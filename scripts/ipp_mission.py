#! /usr/bin/env python3

from aqua_controller import AquaController
from ros_sgp_tools.srv import Waypoints
from ros_sgp_tools.msg import ETA
import numpy as np
import rclpy


class IPPMissionPlanner(AquaController):

    def __init__(self):
        AquaController.__init__(self)

        # Setup current waypoint publisher that publishes at 10Hz
        self.eta_publisher = self.create_publisher(ETA, 'eta', 10)

        # Setup waypoint service
        self.waypoint_service = self.create_service(Waypoints, 
                                                    '/robot_0/waypoints',
                                                    self.waypoint_service_callback)
      
        # Initialize variables
        self.waypoints = None
        self.eta_msg = ETA()
        self.eta_msg.current_waypoint = -1

        self.get_logger().info("Initialized, waiting for waypoints")

        # Wait to get the waypoints from the online IPP planner
        while rclpy.ok() and self.waypoints is None:
            rclpy.spin_once(self, timeout_sec=1.0)

        self.distances = np.linalg.norm(self.waypoints[1:]-self.waypoints[:-1],
                                        axis=-1)

        # Setup timers
        self.eta_timer = self.create_timer(1, self.publish_eta)

        # Start visiting the waypoints
        self.mission()

    def waypoint_service_callback(self, request, response):
        waypoints_msg = request.waypoints.waypoints

        waypoints = []
        for i in range(len(waypoints_msg)):
            waypoints.append([waypoints_msg[i].x, 
                              waypoints_msg[i].y,
                              waypoints_msg[i].z])
        waypoints = np.array(waypoints)

        # Check if the vehicle has already passed some updated waypoints
        if self.waypoints is not None:
            idx = self.eta_msg.current_waypoint
            delta = self.waypoints[:idx+1]-waypoints[:idx+1]
            if np.sum(np.abs(delta)) > 0:
                self.get_logger().info('Waypoints rejected! Vehicle has already passed some updated waypoints')
                self.get_logger().info(f'{delta}')
                response.success = False
                return response
        
        self.waypoints = waypoints
        self.get_logger().info('Waypoints received and accepted')
        response.success = True
        return response
    
    def publish_eta(self):
        idx = self.eta_msg.current_waypoint-1
        if idx < 0:
            return
        self.distances[idx] = self.waypoint_distance
        waypoints_eta = self.distances/self.velocity
        self.eta_msg.eta = []
        for i in range(len(waypoints_eta)):
            eta = -1. if i < idx else np.sum(waypoints_eta[idx:i+1])
            self.eta_msg.eta.append(eta)
        self.eta_publisher.publish(self.eta_msg)

    def mission(self):
        for i in range(len(self.waypoints)):
            self.eta_msg.current_waypoint = i
            self.get_logger().info(f'Visiting waypoint {i}: {self.waypoints[i][:2]}')
            if self.go2waypoint([self.waypoints[i][0],
                                 self.waypoints[i][1]]):
                self.get_logger().info(f'Reached waypoint {i}')
        self.get_logger().info('Mission complete')

def main():
    _ = IPPMissionPlanner()

if __name__ == '__main__':
    main()