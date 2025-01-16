from launch_ros.actions import Node
from launch import LaunchDescription

def generate_launch_description():
    nodes = []
    aipp_service = Node(package='ros_sgp_tools',
                        executable='aipp_service.py',
                        name='aipp_service')
    nodes.append(aipp_service)
    return LaunchDescription(nodes)