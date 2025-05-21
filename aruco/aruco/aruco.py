import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as img
from geometry_msgs.msg import PointStamped

def fit_plane_and_transform(points, reference_id, ids, reference_rvec):
    if len(points) < 4:
        return None

    # Create a dictionary mapping marker ID to its translation vector
    points_dict = {int(ids[i]): np.array(points[i]).flatten() for i in range(len(ids))}

    # Ensure both the reference marker and marker 0 exist
    if reference_id not in points_dict or 0 not in points_dict:
        return None

    # Use marker with ID reference_id as the global origin
    P1 = points_dict[reference_id]

    # Compute the best-fit plane using all markers
    all_points = np.array(list(points_dict.values()))
    mean_point = np.mean(all_points, axis=0)
    centered_points = all_points - mean_point
    _, _, Vt = np.linalg.svd(centered_points)
    # The last row of Vt is the normal (z-axis) of the best-fit plane.
    z_axis = Vt[-1]
    z_axis /= np.linalg.norm(z_axis)

    # Use the reference marker's rotation vector to extract its z-axis.
    R_marker, _ = cv2.Rodrigues(reference_rvec)
    marker_z_axis = R_marker[:, 2]  # Marker z-axis (3rd column)

    # Check the dot product: if positive, flip the computed z_axis.
    if np.dot(z_axis, marker_z_axis) > 0:
        z_axis = -z_axis

    # Use marker with ID 0 to define the y direction, then project it onto the plane
    P0 = points_dict[0]
    raw_y = P0 - P1
    # Remove the component along z_axis to project raw_y into the plane
    raw_y_proj = raw_y - np.dot(raw_y, z_axis) * z_axis
    if np.linalg.norm(raw_y_proj) == 0:
        return None
    y_axis = raw_y_proj / np.linalg.norm(raw_y_proj)

    # The x-axis is the cross product of y and z
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Build the rotation matrix from our computed axes (each axis is a row)
    R = np.vstack([x_axis, y_axis, z_axis])

    def transform_point(P):
        return R @ (P - P1)

    transformed_points = {key: transform_point(val) for key, val in points_dict.items()}
    camera_global= -R @ P1

    return {
        "transformed_points": transformed_points,
        "camera_global": camera_global,
        "origin": np.array([0, 0, 0]),
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis
    }

class ImageSub(Node):
    def __init__(self):
        super().__init__('aruco_pose_estimator')
        self.get_logger().info('ArUco pose estimation node started.')
        self.subscription = self.create_subscription(
            Image, "chaljapls", self.timer_callback, 10)
        self.publisher = self.create_publisher(PointStamped, 'camera_position', 10)
        self.publisher1 = self.create_publisher(PointStamped, 'm1', 10)
        self.publisher2 = self.create_publisher(PointStamped, 'm2', 10)
        self.publisher3 = self.create_publisher(PointStamped, 'm3', 10)
        self.publisher4 = self.create_publisher(PointStamped, 'm0', 10)
        self.br = CvBridge()
        self.reference_id = 2  # Marker ID 2 is the global origin

    def try_detect_markers(self, image):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(np.array(image), aruco_dict, parameters=aruco_params)
        return corners, ids

    def timer_callback(self, data):
        cvimage = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        # Convert BGRA to RGB (or adjust depending on your input image format)
        frame = cv2.cvtColor(cvimage, cv2.COLOR_BGRA2RGB)
        image = img.fromarray(frame)
        corners, ids = self.try_detect_markers(image)

        if ids is not None:
            marker_size = 0.20  # Marker size in meters
            camera_matrix = np.array([[367.164, 0, 478.535],
                                      [0, 367.186, 298.9795],
                                      [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.array([-0.010077, -0.0356293, -0.000188639, 0.000174253, 0.0101471], dtype=np.float32)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            points = []
            detected_ids = []
            for i in range(len(ids)):
                detected_ids.append(int(ids[i][0]))
                points.append(tvecs[i].reshape(3))

            # Process only if exactly 4 markers are detected
            if len(detected_ids) == 4:
                # Find the index for the reference marker (ID = self.reference_id)
                try:
                    reference_index = detected_ids.index(self.reference_id)
                except ValueError:
                    self.get_logger().warn("Reference marker not found in detected IDs.")
                    return

                reference_rvec = rvecs[reference_index]
                result = fit_plane_and_transform(points, self.reference_id, detected_ids, reference_rvec)
                if result:
                    transformed_points = result["transformed_points"]
                    cam_pos = result["camera_global"]
                    x_axis = result["x_axis"]
                    y_axis = result["y_axis"]
                    z_axis = result["z_axis"]

                    # Log marker IDs with their transformed coordinates and the computed axes
                    for m_id, t_coord in transformed_points.items():
                        self.get_logger().info(f"Marker {m_id} transformed: {t_coord}")

      
                    self.get_logger().info(f"X-axis: {x_axis}, Y-axis: {y_axis}, Z-axis: {z_axis}")

                    msg = PointStamped()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "map"
                    msg.point.x = float(cam_pos[0])
                    msg.point.y = float(cam_pos[1])
                    msg.point.z = float(cam_pos[2])
                    self.publisher.publish(msg)
                    msg.point.x = float(transformed_points[1][0])
                    msg.point.y = float(transformed_points[1][1])
                    msg.point.z = float(transformed_points[1][2])
                    self.publisher1.publish(msg)
                    msg.point.x = float(transformed_points[2][0])
                    msg.point.y = float(transformed_points[2][1])
                    msg.point.z = float(transformed_points[2][2])
                    self.publisher2.publish(msg)
                    msg.point.x = float(transformed_points[3][0])
                    msg.point.y = float(transformed_points[3][1])
                    msg.point.z = float(transformed_points[3][2])
                    self.publisher3.publish(msg)
                    msg.point.x = float(transformed_points[0][0])
                    msg.point.y = float(transformed_points[0][1])
                    msg.point.z = float(transformed_points[0][2])
                    self.publisher4.publish(msg)
        else:
            self.get_logger().warn("No ArUco markers detected.")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
