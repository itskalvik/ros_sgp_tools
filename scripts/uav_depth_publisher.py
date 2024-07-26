#! /usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix, Range, Image
import numpy as np

from cv_bridge import CvBridge
import cv2

class DepthPublisher(Node):

    def __init__(self):
        super().__init__('depth_publisher')
        # Setup the depth publisher
        self.depth_publisher = self.create_publisher(Range, 
                                'mavros/rangefinder/rangefinder', 
                                10)
        self.depth_msg = Range()

        # Setup the position subscriber
        self.vehicle_pose_subscriber = self.create_subscription(Image, 
                                            'zed/zed_node/depth/depth_registered', 
                                            self.data_callback, 
                                            rclpy.qos.QoSProfile(depth=10))
        self.get_logger().info("Initialized depth publisher")

        # Setup variables to get data from depth map
        self.bridge = CvBridge()

        # Get data from 3x3 grid
        delta = 1280//6
        mask_x = [delta, delta*3, delta*5]
        delta = 720//6
        mask_y = [delta, delta*3, delta*5]
        mask_x, mask_y = np.meshgrid(mask_x, mask_y)
        self.mask_x = mask_x.reshape(-1)
        self.mask_y = mask_y.reshape(-1)
        
        self.bool_mask = np.array([[-1,  1], [0,  1], [1,  1],
                                   [-1,  0], [0,  0], [1,  0],
                                   [-1, -1], [0, -1], [1, -1]])

        # ZED FoV 110° (H) x 70° (V)
        # Mapped to right triangle base for 1 meter height
        self.fov_scale = np.array([1.42815, 0.70021])
        self.dist_scale = 100.0

    def data_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        data_X, data_y = [], []
        height = np.mean(depth[np.where(np.isfinite(depth))])
        self.get_logger().info(f'{height}')
        for i, point in enumerate(zip(self.mask_x, self.mask_y)):
            data = depth[point[1], point[0]]
            if np.isfinite(data):
                depth = cv2.circle(depth, np.array((self.mask_x[4], self.mask_y[4]) + \
                                                   (self.bool_mask[i]*height*self.fov_scale*self.dist_scale)).astype(int), 
                                   radius=10, 
                                   color=(0, 200, 2), 
                                   thickness=-1)

        cv2.imshow("Image window", depth)
        cv2.waitKey(3)


def main(args=None):
    rclpy.init(args=args)
    depth_publisher = DepthPublisher()
    rclpy.spin(depth_publisher)
    depth_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
