import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2


class BgraToBgrConverter(Node):
    def __init__(self):
        super().__init__('bgra_to_bgr_converter')
        self.bridge = CvBridge()
        
        # Initialize subscriber and publisher
        self.pub = self.create_publisher(Image, 'set1', 10)
        self.sub = self.create_subscription(Image, 'set', self.image_callback, 'raw', 10)

    def image_callback(self, img_msg, info_msg):
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgra8')
        except cv2.error as e:
            self.get_logger().error('CvBridge error: %s' % str(e))
            return

        # Convert BGRA to BGR
        bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)

        # Convert OpenCV image back to ROS image message
        out_img_msg = self.bridge.cv2_to_imgmsg(bgr_image, encoding='bgr8')
        out_img_msg.header = img_msg.header

        # Publish the new image and camera info
        self.pub.publish(out_img_msg, info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BgraToBgrConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

