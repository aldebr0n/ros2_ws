import rclpy
from rclpy.node import Node

from graphing.msg import Xycord                      
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('sub')
        self.subscription = self.create_subscription(Xycord, 'topic', self.listener_callback, 4)
        

    def listener_callback(self, msg):
        
        xcord = msg.xcord
        ycord = msg.ycord
        pointscolor = msg.pointscolor
        points = []
        plt.show(block=False)
        plt.cla()
        
        
        for i in range(len(pointscolor)):
            points.append([xcord[i], ycord[i]])
        if (len(points)<4):
            return 
        else:    
            points = np.array(points)
            tri = Delaunay(points)
            points = points.tolist()

        for i in tri.simplices:
            i = i.tolist()
            
            x1 = np.array(points[i[0]][0])
            y1 = np.array(points[i[0]][1])
            x2 = np.array(points[i[1]][0])
            y2 = np.array(points[i[1]][1])
            x3 = np.array(points[i[2]][0])
            y3 = np.array(points[i[2]][1])
            
            if (pointscolor[i[0]] == pointscolor[i[1]] and pointscolor[i[1]] == pointscolor[i[2]]):
                continue

            if pointscolor[i[0]] == 0:
                plt.plot(x1, y1, 'bo')
            else:
                plt.plot(x1, y1, 'yo')
                
            if pointscolor[i[1]] == 0:
                plt.plot(x2, y2, 'bo')
            else:
                plt.plot(x2, y2, 'yo')
                
            if pointscolor[i[2]] == 0:
                plt.plot(x3, y3, 'bo')
            else:
                plt.plot(x3, y3, 'yo')
                
            xpt1 = np.array([points[i[0]][0], points[i[1]][0]])
            ypt1 = np.array([points[i[0]][1], points[i[1]][1]])
            xpt2 = np.array([points[i[1]][0], points[i[2]][0]])
            ypt2 = np.array([points[i[1]][1], points[i[2]][1]])
            xpt3 = np.array([points[i[0]][0], points[i[2]][0]])
            ypt3 = np.array([points[i[0]][1], points[i[2]][1]])
            
            plt.plot(xpt1, ypt1, 'k')
            plt.plot(xpt2, ypt2, 'k')
            plt.plot(xpt3, ypt3, 'k')

            if pointscolor[i[0]] == pointscolor[i[1]]:
                xp = np.array([(points[i[0]][0] + points[i[2]][0]) * 0.5])
                yp = np.array([(points[i[0]][1] + points[i[2]][1]) * 0.5])
                plt.plot(xp, yp, 'ro')
                xp = np.array([(points[i[2]][0] + points[i[1]][0]) * 0.5])
                yp = np.array([(points[i[2]][1] + points[i[1]][1]) * 0.5])
                plt.plot(xp, yp, 'ro')
                xp = np.array([(points[i[0]][0] + points[i[2]][0]) * 0.5, (points[i[1]][0] + points[i[2]][0]) * 0.5])
                yp = np.array([(points[i[0]][1] + points[i[2]][1]) * 0.5, (points[i[1]][1] + points[i[2]][1]) * 0.5])
                plt.plot(xp, yp, 'g')

            elif pointscolor[i[2]] == pointscolor[i[1]]:
                xp = np.array([(points[i[0]][0] + points[i[2]][0]) * 0.5])
                yp = np.array([(points[i[0]][1] + points[i[2]][1]) * 0.5])
                plt.plot(xp, yp, 'ro')
                xp = np.array([(points[i[1]][0] + points[i[0]][0]) * 0.5])
                yp = np.array([(points[i[1]][1] + points[i[0]][1]) * 0.5])
                plt.plot(xp, yp, 'ro')
                xp = np.array([(points[i[0]][0] + points[i[2]][0]) * 0.5, (points[i[0]][0] + points[i[1]][0]) * 0.5])
                yp = np.array([(points[i[0]][1] + points[i[2]][1]) * 0.5, (points[i[0]][1] + points[i[1]][1]) * 0.5])
                plt.plot(xp, yp, 'g')
            else:
                xp = np.array([(points[i[0]][0] + points[i[1]][0]) * 0.5])
                yp = np.array([(points[i[0]][1] + points[i[1]][1]) * 0.5])
                plt.plot(xp, yp, 'ro')
                xp = np.array([(points[i[2]][0] + points[i[1]][0]) * 0.5])
                yp = np.array([(points[i[2]][1] + points[i[1]][1]) * 0.5])
                plt.plot(xp, yp, 'ro')
                xp = np.array([(points[i[0]][0] + points[i[1]][0]) * 0.5, (points[i[1]][0] + points[i[2]][0]) * 0.5])
                yp = np.array([(points[i[0]][1] + points[i[1]][1]) * 0.5, (points[i[1]][1] + points[i[2]][1]) * 0.5])
                plt.plot(xp, yp, 'g')

        plt.show()
        

def main(args=None):
    rclpy.init(args=args)
    sub = MinimalSubscriber()
    rclpy.spin(sub)
    sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

