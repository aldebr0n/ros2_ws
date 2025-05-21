import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image as img

global model
model = YOLO('/home/afr/ros2_ws/src/testimage/testimage/bestest.pt')

class SimplePubSub(Node):
    def __init__(self):
        super().__init__('simple_pub_sub')

        topic_name = 'video_frames'

        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.timer = self.create_timer(0, self.timer_callback)

        self.cap = cv2.VideoCapture(0)
        self.br = CvBridge()

        self.subscription = self.create_subscription(Image, topic_name, self.img_callback, 10)
        self.subscription 
        self.br = CvBridge()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            self.publisher_.publish(self.br.cv2_to_imgmsg(frame))
        self.get_logger().info('Publishing video frame')

    def img_callback(self, data):
        self.get_logger().info('starting')

        self.get_logger().info('after yolo')

        transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])

        self.get_logger().info('after transform')

        current_frame = self.br.imgmsg_to_cv2(data)
        frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PIL Image
        image = img.fromarray(frame)

        imagealmostready = transform(image)
        imageready = imagealmostready.unsqueeze(0)
        imageready = imageready.float()
        imageready = imageready.to('cuda')

        self.get_logger().info('after conversions')

        with torch.no_grad():
            results = model(imageready)

        self.get_logger().info('inference')

        for prediction in results:
            boxtensor = prediction.boxes.xywh
            boxes = boxtensor.cpu().numpy()  # Convert tensor to numpy array

            for box in boxes:
                x = int(box[0])
                y = int(box[1] * 0.75)
                x_plus_w = int(box[2])
                y_plus_h = int(box[3] * 0.75)
                cv2.rectangle(current_frame, (x - int(x_plus_w / 2), y + int(y_plus_h / 2)),
                              (x + int(x_plus_w / 2), y - int(y_plus_h / 2)), (0, 0, 0))
                cv2.putText(current_frame, "cones", (x - int(x_plus_w / 2), y + int(y_plus_h / 2) - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 2)

        cv2.imshow("camera", current_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)

    simple_pub_sub = SimplePubSub()
    rclpy.spin(simple_pub_sub)
    simple_pub_sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

