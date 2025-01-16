#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from subprocess import call
from std_srvs.srv import Empty


class AIPPService(Node):

    def __init__(self):
        super().__init__('aipp_service')
        self.srv = self.create_service(Empty, 'aipp_service', 
                                       self.callback)

    def callback(self, request, response):
        self.get_logger().info('Starting AIPP Method')
        f = open("aipp_service.log", "w")
        call(["ros2", "launch", 
              "ros_sgp_tools", 
              "single_robot.launch.py"], 
              stdout=f)
        return response


def main(args=None):
    rclpy.init(args=args)
    service = AIPPService()
    rclpy.spin(service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
