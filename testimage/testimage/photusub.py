import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image as img
from graphing.msg import Xycord
from scipy.spatial import Delaunay
import pyzed.sl as sl
from visualization_msgs.msg import Marker
import math

global model
model = YOLO('/home/afr/ros2_ws/src/testimage/testimage/bestest.pt')

# Create a ZED camera object
#zed = sl.Camera()

# Set configuration parameters
#init_params = sl.InitParameters()
#init_params.camera_resolution = sl.RESOLUTION.SVGA  # Use HD720 or HD1200 video mode, depending on camera type.
#init_params.camera_fps = 30  # Set fps at 30

# Open the camera
#err1 = zed.open(init_params)
global xc
global yc
global zc
class ImageSub(Node):
    def __init__(self):
        super().__init__('simple_pub_sub')
        self.get_logger().info('sub starting')

        #self.timer = self.create_timer(0, self.timer_callback)
        self.publisher_ = self.create_publisher(Xycord, 'topic', 10)
        self.publisher2_ =self.create_publisher(Marker ,'cone_placed' , 10)
        self.subscription = self.create_subscription(Image, "chaljapls", self.timer_callback, 10)
        self.subscription
        self.br = CvBridge()
        self.xcord = []
        self.ycord = []
        self.pointscolor = []
        
    	

    def timer_callback(self, data):
        msg = Xycord()
        msg2= Marker()
        msg2.header.frame_id="/map"
        msg2.header.stamp = self.get_clock().now().to_msg()
        msg2.type=2
        msg2.id=0
        msg2.scale.x=1.0
        msg2.scale.y=1.0
        msg2.scale.z=1.0
        self.get_logger().info('starting')

        transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
        self.get_logger().info('after transform')

        #if err1 != sl.ERROR_CODE.SUCCESS:
            #exit(-1)
        #image = sl.Mat()
        #point_cloud=sl.Mat()
        #runtime_parameters = sl.RuntimeParameters()

        # Grab an image, a RuntimeParameters object must be given to grab()
        #if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            #zed.retrieve_image(image, sl.VIEW.LEFT)
            #zed.retrieve_measure(point_cloud,sl.MEASURE.XYZRGBA)
            
            #image_ocv = image.get_data()

        cvimage = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')

        self.get_logger().info('after transform')
        frame = cv2.cvtColor(cvimage, cv2.COLOR_BGRA2RGB)

        image = img.fromarray(frame)

        image_almost_ready = transform(image)
        image_ready = image_almost_ready.unsqueeze(0)
        image_ready = image_ready.float()
        image_ready = image_ready.to('cuda')

        self.get_logger().info('after conversions')

        with torch.no_grad():
            results = model(image_ready)

        self.get_logger().info('inference')

        for prediction in results:
            boxtensor = prediction.boxes.xywh
            dabbe = boxtensor.cpu().numpy()
            number = prediction.boxes.cls.cpu().numpy()
            confi = prediction.boxes.conf.cpu().numpy()
            i = 0
            self.xcord = []
            self.ycord = []
            self.pointscolor = []
            for box in dabbe:
                x = int(box[0] * 1.5)
                y = int(box[1] * 0.9375)
                self.xcord.append(x)
                self.ycord.append(y)
                #err,point_cloud_value=point_cloud.get_value(x,y)
                #if math.isfinite(point_cloud_value[2]):
                	#msg2.pose.position.x=float(point_cloud_value[0])
                	#msg2.pose.position.y=float(point_cloud_value[1])
                	#msg2.pose.position.z=float(point_cloud_value[2])
                #else:
                	#msg2.pose.position.x=0.0
                	#msg2.pose.position.y=0.0
                	#msg2.pose.position.z=0.0
                	
                msg2.pose.orientation.x=0.0
                msg2.pose.orientation.y=0.0
                msg2.pose.orientation.z=0.0
                msg2.pose.orientation.w=1.0
                msg2.pose.position.x=0.0
                msg2.pose.position.y=0.0
                msg2.pose.position.z=0.0
                
                x_plus_w = int(box[2] * 1.5)
                y_plus_h = int(box[3] * 0.9375)
                cv2.rectangle(frame, (x - int(x_plus_w / 2), y + int(y_plus_h / 2)),
                              (x + int(x_plus_w / 2), y - int(y_plus_h / 2)), (0, 0, 0))

                if number[i] == 0:
                    cv2.putText(frame, "BLUE CONE", (x - int(x_plus_w / 2), y + int(y_plus_h / 2) - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    self.pointscolor.append(0)
                    msg2.color.r=0.0
                    msg2.color.g=0.0
                    msg2.color.b=1.0
                    msg2.color.a=1.0
                elif number[i] == 1:
                    self.pointscolor.append(2)
                    msg2.color.r=1.0
                    msg2.color.g=0.7
                    msg2.color.b=0.0
                    msg2.color.a=1.0
                    cv2.putText(frame, "ORANGE CONE", (x - int(x_plus_w / 2), y + int(y_plus_h / 2) - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                               
                else:
                    cv2.putText(frame, "YELLOW CONE", (x - int(x_plus_w / 2), y + int(y_plus_h / 2) - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    self.pointscolor.append(1)
                    msg2.color.r=1.0
                    msg2.color.g=1.0
                    msg2.color.b=0.0
                    msg2.color.a=1.0
                
                framedis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("camera", framedis)
                cv2.waitKey(1)
                i += 1

        self.get_logger().info('making boxes')
        msg.xcord = self.xcord
        msg.ycord = self.ycord
        msg.pointscolor = self.pointscolor
        self.publisher_.publish(msg)
        self.publisher2_.publish(msg2)

def main(args=None):
    rclpy.init(args=args)

    simple_pub_sub = ImageSub()
    rclpy.spin(simple_pub_sub)
    zed.close()
    simple_pub_sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

