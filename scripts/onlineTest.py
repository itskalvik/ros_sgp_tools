#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ros_sgp_tools.msg import IPPData
from ros_sgp_tools.srv import IPP
from geometry_msgs.msg import Point

class OfflineIPPService(Node):
    def __init__(self):
        super().__init__('offline_ipp_service')
        self.srv = self.create_service(IPP, 'tb3_0/offlineIPP', self.handle_offline_ipp)
        self.get_logger().info('OfflineIPP Service is ready')

    def handle_offline_ipp(self, request, response):
        self.get_logger().info(f'Received waypoints: {request.data.waypoints}')
        self.get_logger().info(f'Received x_train: {request.data.x_train}')

        response.success = True

        return response

def main(args=None):
    rclpy.init(args=args)
    offline_ipp_service = OfflineIPPService()
    rclpy.spin(offline_ipp_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

