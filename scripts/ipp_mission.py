#! /usr/bin/env python3

from ros_sgp_tools.srv import Waypoints
from std_msgs.msg import Int32

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.executors import MultiThreadedExecutor

from guided_mission import MissionPlanner
from time import sleep


class IPPMissionPlanner(MissionPlanner):

    def __init__(self):
        super().__init__()

        # Setup current waypoint publisher that publishes at 10Hz
        self.current_waypoint_publisher = self.create_publisher(Int32, 'current_waypoint', 10)
        self.current_waypoint_timer = self.create_timer(1/10, self.publish_current_waypoint)
  
        # Setup waypoint service
        self.waypoint_service = self.create_service(Waypoints, '/waypoints',
                                                    self.waypoint_service_callback)
      
        # Initialize variables
        self.waypoints = [] 
        self.current_waypoint = -1

        self.get_logger().info("Initialized, waiting for waypoints")

    def waypoint_service_callback(self, request, response):
        waypoints = request.waypoints.waypoints
        self.waypoints = []
        for i in range(len(waypoints)):
            self.waypoints.append([waypoints[i].x, waypoints[i].y])
        self.get_logger().info('Waypoints received')
        response.current_waypoint = self.current_waypoint
        return response

    def publish_current_waypoint(self):
        current_waypoint_msg = Int32()
        current_waypoint_msg.data = self.current_waypoint
        self.current_waypoint_publisher.publish(current_waypoint_msg)    

    def mission(self):
        "GUIDED mission"

        sleep(5) # Wait to get the state of the vehicle

        if self.arm(True):
            self.get_logger().info('Armed')

        if self.engage_mode('GUIDED'):
            self.get_logger().info('GUIDED mode Engaged')

        self.current_waypoint = 0
        if self.go2waypoint([35.30684387683425, -80.7360063599907]):
            self.get_logger().info('Reached waypoint')
            self.current_waypoint += 1

        if self.go2waypoint([35.30684275566786, -80.73612370299257]):
            self.get_logger().info('Reached waypoint')
            self.current_waypoint += 1

        if self.go2waypoint([35.30679876645213, -80.73623439122146]):
            self.get_logger().info('Reached waypoint')
            self.current_waypoint += 1

        if self.go2waypoint([35.30674267884529, -80.73600329951549]):
            self.get_logger().info('Reached waypoint')
            self.current_waypoint += 1

        if self.arm(False):
            self.get_logger().info('Disarmed')

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    try:
        mission_planner = IPPMissionPlanner()

        executor = MultiThreadedExecutor()
        executor.add_node(mission_planner)
        executor.create_task(mission_planner.mission)
        executor.spin()

    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        rclpy.shutdown()
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
