#! /usr/bin/env python3
import time
import rclpy
import tf_transformations
from rclpy.node import Node
from collections import deque
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from aqua2_interfaces.msg import AutopilotCommand
from aqua2_interfaces.srv import GetBool, SetString, SetInt
from geometry_msgs.msg import PoseWithCovarianceStamped

from controllers import *


class AquaController(Node):

    def __init__(self):
        super().__init__('aqua_controller')

        # Reset DVL
        self.get_logger().info('Resetting DVL')
        reset_odometry = self.create_client(Empty, '/aqua/dvl/reset_odometry')
        response = self.call_client(reset_odometry, Empty.Request())
        assert response.success
        self.get_logger().info('--> DVL reset complete')

        # Reset IMU
        self.get_logger().info('Resetting IMU')
        zero_heading = self.create_client(Empty, '/aqua/imu/zero_heading')
        response = self.call_client(zero_heading, Empty.Request())
        assert response.success
        self.get_logger().info(f'--> IMU reset complete')

        # Reset heading
        self.get_logger().info('Resetting pose')
        set_pose = self.create_client(Empty, '/aqua/set_pose')
        response = self.call_client(set_pose, Empty.Request())
        self.get_logger().info(f'--> Pose reset complete ({response})')

        # Make sure the system is calibrated
        is_calibrated = self.create_client(GetBool, '/aqua/system/is_calibrated')
        calibrate = self.create_client(Empty, '/aqua/system/calibrate')
        while not self.call_client(is_calibrated, GetBool.Request()).value:
            self.get_logger().info(f'Calibrating')
            self.call_client(calibrate, Empty.Request())
            time.sleep(1)
        self.get_logger().info('--> Calibration complete!')

        # Enable swim mode
        self.get_logger().info('Activating swim mode')
        set_mode = self.create_client(SetString, '/aqua/system/set_mode')
        response = self.call_client(set_mode, SetString.Request(value='swimmode'))
        assert "Switching" in response.msg or "already" in response.msg, response.msg
        self.get_logger().info('--> Swim mode activated!')

        time.sleep(2)

        # Activate the depth-based autopilot
        self.get_logger().info('Switching to autopilot depth mode.')
        set_autopilot_mode = self.create_client(SetInt, '/aqua/autopilot/set_autopilot_mode')
        response = self.call_client(set_autopilot_mode, SetInt.Request(value=4))
        assert response.msg == ''
        self.get_logger().info('--> Autopilot activated.')

        # Init vars
        self.mission_depth = 3.0
        self.vehicle_position = np.array([0., 0., 0.])
        self.autopilot_command = AutopilotCommand()
        self.autopilot_command.target_depth = self.mission_depth
        self.use_altitude = False
        self.angle_tolerance = 0.1
        self.position_tolerance = 0.5
        self.velocity = 0.01
        self.velocity_buffer = deque([0.01])
        self.waypoint_distance = -1

        # Setup subscribers
        # SENSOR_QOS used for most of sensor streams
        SENSOR_QOS = rclpy.qos.qos_profile_sensor_data
        # STATE_QOS used for state topics, like ~/state, ~/mission/waypoints etc.
        STATE_QOS = rclpy.qos.QoSProfile(
            depth=10, 
            durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.vehicle_pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped, '/aqua/dvl_pose_estimate',
            self.vehicle_pose_callback, SENSOR_QOS)
        
        self.vehicle_odom_subscriber = self.create_subscription(
            Odometry, '/aqua/simulator/pose',
            self.vehicle_odom_callback, SENSOR_QOS)

        # Create publishers
        self.autopilot_command_publisher = self.create_publisher(
            AutopilotCommand, '/aqua/autopilot/command', STATE_QOS)
    
        # Create the trajectory controller
        self.get_control_cmd = create_hybrid_unicycle_pose_controller(linear_velocity_gain=8.0,
                                                                      angular_velocity_gain=0.6,
                                                                      position_epsilon=0.3, 
                                                                      position_error=self.position_tolerance, 
                                                                      rotation_error=self.angle_tolerance)
        
    def call_client(self, cli, request):
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        future = cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def vehicle_odom_callback(self, msg):
        """Callback function for vehicle odom topic subscriber.
        Computes nominal linear velocity"""
        velocity = np.hypot(msg.twist.twist.linear.x,
                            msg.twist.twist.linear.y)
        if len(self.velocity_buffer) > 50:
            self.velocity_buffer.popleft()
        if velocity > 0.2:
            self.velocity_buffer.append(velocity)
        self.velocity = np.mean(self.velocity_buffer)

    def vehicle_pose_callback(self, msg):
        self.vehicle_position[0] = msg.pose.pose.position.x
        self.vehicle_position[1] = msg.pose.pose.position.y

        q = []
        q.append(msg.pose.pose.orientation.x)
        q.append(msg.pose.pose.orientation.y)
        q.append(msg.pose.pose.orientation.z)
        q.append(msg.pose.pose.orientation.w)
        (_, _, yaw) = tf_transformations.euler_from_quaternion(q)
        self.vehicle_position[2] = yaw

    def at_waypoint(self, waypoint, xy_tolerance=0.7):
        """Check if the vehicle is at the waypoint."""
        dist = self.vehicle_position[:2].reshape(1, -1)-np.array(waypoint[:2]).reshape(1, -1)
        self.waypoint_distance = np.linalg.norm(dist)
        if self.waypoint_distance < self.position_tolerance:
            return True
        else:
            return False

    def go2waypoint(self, goal):
        goal.append(0.0)
        rotation_complete = False
        while not self.at_waypoint(goal) and rclpy.ok():
            # Compute best approach angle to goal
            goal[2] = np.arctan2(goal[1] - self.vehicle_position[1], goal[0] - self.vehicle_position[0])

            # Rotate the robot towards the goal first
            if not rotation_complete:
                error_angle = goal[2] - self.vehicle_position[2]
                # Wrap the angle to [-π, π]
                error_angle = np.arctan2(np.sin(error_angle), np.cos(error_angle))
                if np.abs(error_angle) > self.angle_tolerance:
                    control_cmd = [[0.], [goal[2]]]
                else:
                    rotation_complete = True
                    self.get_logger().info(f'Rotation complete')
                    continue
            # Move the robot to the goal
            else:
                control_cmd = self.get_control_cmd(np.array(self.vehicle_position).reshape(-1, 1), 
                                                   np.array(goal).reshape(-1, 1))

            # Publish the control command
            if control_cmd[0][0] < 0:
                rotation_complete = False
                self.get_logger().info(f'Realigning with the waypoint')
                continue

            # Ignore unicycle model controller for yaw control
            target_yaw = np.arctan2(np.sin(goal[2]), np.cos(goal[2]))
            target_yaw = np.degrees(target_yaw)

            self.autopilot_command.target_yaw = target_yaw
            self.autopilot_command.surge = np.clip(control_cmd[0][0], 0., 1.)
            self.autopilot_command_publisher.publish(self.autopilot_command)
            rclpy.spin_once(self)

        # Stop the robot
        self.autopilot_command.surge = 0.0
        self.autopilot_command.target_yaw = 0.0
        self.autopilot_command_publisher.publish(self.autopilot_command)
        rclpy.spin_once(self)
        return True

    def mission(self):
        self.get_logger().info('Visiting waypoint 1')
        if self.go2waypoint([5.0, 0.0]):
            self.get_logger().info('Reached waypoint 1')

        self.get_logger().info('Visiting waypoint 2')
        if self.go2waypoint([5.0, 5.0]):
            self.get_logger().info('Reached waypoint 3')

        self.get_logger().info('Visiting waypoint 3')
        if self.go2waypoint([0.0, 5.0]):
            self.get_logger().info('Reached waypoint 3')

        self.get_logger().info('Visiting waypoint 4')
        if self.go2waypoint([0.0, 0.0]):
            self.get_logger().info('Reached waypoint 4')

def main(args=None):
    rclpy.init(args=args)
    node = AquaController()
    node.mission()

if __name__ == '__main__':
    main()