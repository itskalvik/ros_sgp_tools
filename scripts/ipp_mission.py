#! /usr/bin/env python3

from ros_sgp_tools.srv import Waypoints
from std_msgs.msg import Int32

import rclpy
from rclpy.executors import ExternalShutdownException

from guided_mission import MissionPlanner


class IPPMissionPlanner(MissionPlanner):

    def __init__(self):
        super().__init__()

        # Setup current waypoint publisher that publishes at 10Hz
        self.current_waypoint_publisher = self.create_publisher(Int32, 'current_waypoint', 10)
        self.current_waypoint_timer = self.create_timer(1/10, self.publish_current_waypoint)
  
        # Setup waypoint service
        self.waypoint_service = self.create_service(Waypoints, 
                                                    'waypoints',
                                                    self.waypoint_service_callback)
      
        # Initialize variables
        self.waypoints = None
        self.current_waypoint = Int32()
        self.current_waypoint.data = -1

        self.get_logger().info("Initialized, waiting for waypoints")

        # Wait to get the waypoints from the online IPP planner
        while rclpy.ok() and self.waypoints is None:
            rclpy.spin_once(self, timeout_sec=1.0)

        # Start visiting the waypoints
        self.mission()

    def waypoint_service_callback(self, request, response):
        waypoints = request.waypoints.waypoints

        self.waypoints = []
        for i in range(len(waypoints)):
            self.waypoints.append([waypoints[i].x, waypoints[i].y])

        self.get_logger().info('Waypoints received')
        response.success = True
        return response
    
    def publish_current_waypoint(self):
        self.current_waypoint_publisher.publish(self.current_waypoint)

    def mission(self):
        """IPP mission"""

        self.get_logger().info('Engaging GUIDED mode')
        if self.engage_mode('GUIDED'):
            self.get_logger().info('GUIDED mode Engaged')

        self.get_logger().info('Arming')
        if self.arm(True):
            self.get_logger().info('Armed')

        for i in range(len(self.waypoints)):
            self.current_waypoint.data = i+1
            self.get_logger().info(f'Visiting waypoint {i+1}')
            if self.go2waypoint([self.waypoints[i][1], self.waypoints[i][0]]):
                self.get_logger().info(f'Reached waypoint {i+1}')

        if self.arm(False):
            self.get_logger().info('Disarmed')

        self.get_logger().info('Mission complete')


def main(args=None):
    rclpy.init(args=args)

    try:
        mission_planner = IPPMissionPlanner()
        rclpy.spin_once(mission_planner)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        mission_planner.destroy_node()
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
