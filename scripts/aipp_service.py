#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
import subprocess
from signal import SIGINT
from time import gmtime, strftime


# used to start and stop launch files
class RLaunch:
    def __init__(self, log_path=''):
        self.process = None

        if not len(log_path):
            self.stdout = subprocess.STDOUT
            self.stderr = subprocess.STDOUT
            self.stdin = subprocess.PIPE
        else:
            log_file = open(log_path, 'a')
            self.stdout = log_file
            self.stderr = log_file
            self.stdin = subprocess.DEVNULL

    # start a launch file
    def start(self, package_name: str, file_name: str, args: list[str] = []) -> None:
        if self.process:
            self.stop()

        args = ['ros2', 'launch', package_name, file_name] + args
        self.process = subprocess.Popen(args, stdout=self.stdout, stderr=self.stderr, stdin=self.stdin)


    # stop a launch file
    def stop(self):
        if not self.process:
            return

        self.process.send_signal(SIGINT)
        self.process.wait()

        self.process = None

class AIPPService(Node):

    def __init__(self):
        super().__init__('aipp_service')
        self.srv = self.create_service(Empty, 'aipp_service', 
                                       self.callback)
        self.started = False
        fname = f"/home/aqua/IndependentRobotics/bags/IPP_mission_log-{strftime('%H-%M-%S', gmtime())}"
        self.r_launch = RLaunch(log_path='/home/aqua/IndependentRobotics/bags/launch.log')

    def callback(self, request, response):
        self.get_logger().info('Starting AIPP Method')

        if not self.started:
            self.r_launch.start(package_name='ros_sgp_tools',
                                file_name='single_robot.launch.py')
            self.started = True
        else:            
            self.r_launch.stop()
            self.started = False
        return response


def main(args=None):
    rclpy.init(args=args)
    service = AIPPService()
    rclpy.spin(service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
