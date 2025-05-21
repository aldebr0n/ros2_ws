import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2 

    

class imagepub(Node):

    def __init__(self):
        super().__init__('imagepub')
        topic_name='video_frames'
        self.publisher_ = self.create_publisher(Image, topic_name , 10)
        self.timer = self.create_timer(0, self.timer_callback)

        self.cap = cv2.VideoCapture(0)
        self.br = CvBridge()
    def timer_callback(self):
        ret, frame = self.cap.read()     
        if ret == True:
            self.publisher_.publish(self.br.cv2_to_imgmsg(frame))
            self.get_logger().info('published')
def main(args=None):
    rclpy.init(args=args)
   
    simple_pub =imagepub()
    rclpy.spin(simple_pub)
    simple_pub.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
