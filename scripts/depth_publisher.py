#! /usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix, Range


class DepthPublisher(Node):

    def __init__(self):
        super().__init__('depth_publisher')
        # Setup the depth publisher
        self.depth_publisher = self.create_publisher(Range, 
                                'mavros/rangefinder_pub', 
                                10)
        self.depth_msg = Range()

        # Setup the position subscriber
        self.vehicle_pose_subscriber = self.create_subscription(NavSatFix, 
                                            'mavros/global_position/global', 
                                            self.data_callback, 
                                            rclpy.qos.qos_profile_sensor_data)
        self.get_logger().info("Initialized depth publisher")

    def data_callback(self, msg):
        self.depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.depth_msg.range = msg.altitude
        self.depth_publisher.publish(self.depth_msg)


def main(args=None):
    rclpy.init(args=args)
    depth_publisher = DepthPublisher()
    rclpy.spin(depth_publisher)
    depth_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
