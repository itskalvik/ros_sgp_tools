#! /usr/bin/env python3

from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from aqua_controller import AquaController
from ros_sgp_tools.srv import Waypoints
from ros_sgp_tools.msg import ETA
from threading import Thread
from rclpy.node import Node
import numpy as np
import rclpy


class WaypointService(Node):
    def __init__(self, mission_node):
        super().__init__('waypoint_service')

        # Mission node
        self.mission_node = mission_node

        # Setup waypoint service
        self.waypoint_service = self.create_service(Waypoints, 
                                                    '/robot_0/waypoints',
                                                    self.waypoint_service_callback)

    def waypoint_service_callback(self, request, response):
        waypoints_msg = request.waypoints.waypoints

        waypoints = []
        for i in range(len(waypoints_msg)):
            waypoints.append([waypoints_msg[i].x, 
                              waypoints_msg[i].y,
                              waypoints_msg[i].z])
        waypoints = np.array(waypoints)

        # Check if the vehicle has already passed some updated waypoints
        if self.mission_node.waypoints is not None:
            idx = self.mission_node.eta_msg.current_waypoint
            delta = self.mission_node.waypoints[:idx+1]-waypoints[:idx+1]
            if np.sum(np.abs(delta)) > 0:
                self.get_logger().info('Waypoints rejected! Vehicle has already passed some updated waypoints')
                self.get_logger().info(f'{delta}')
                response.success = False
                return response
        
        self.mission_node.distances = np.linalg.norm(waypoints[1:]-waypoints[:-1],
                                                     axis=-1)
        self.mission_node.waypoints = waypoints
        self.get_logger().info('Waypoints received and accepted')
        response.success = True
        return response

class IPPMissionPlanner(AquaController):

    def __init__(self):
        AquaController.__init__(self)

        # Initialize variables
        self.waypoints = None
        self.eta_msg = ETA()
        self.eta_msg.current_waypoint = -1
    
        # Setup current waypoint publisher that publishes at 10Hz
        self.eta_publisher = self.create_publisher(ETA, '/robot_0/eta', 10)

        # Setup timers
        self.eta_timer = self.create_timer(1, self.publish_eta)

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
        while rclpy.ok() and self.waypoints is None:
            rclpy.spin_once(self, timeout_sec=1.0)

        for i in range(len(self.waypoints)):
            self.eta_msg.current_waypoint = i
            self.go2waypoint(np.round(self.waypoints[i, :2]))

        self.get_logger().info('Mission complete')

def spin_srv(executor):
    try:
        executor.spin()
    except rclpy.executors.ExternalShutdownException:
        pass


def main():
    mission_node = IPPMissionPlanner()
    node_executor = MultiThreadedExecutor()
    node_executor.add_node(mission_node)
    
    service_node = WaypointService(mission_node)
    srv_executor = SingleThreadedExecutor()
    srv_executor.add_node(service_node)
    srv_thread = Thread(target=spin_srv, args=(srv_executor, ), daemon=True)
    srv_thread.start()

    mission_node.mission()

if __name__ == '__main__':
    main()