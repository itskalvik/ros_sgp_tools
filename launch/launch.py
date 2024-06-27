from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros_sgp_tools',
            executable='offline_ipp.py',
            name='offline_ipp'
        ),
        
        Node(
            package='ros_sgp_tools',
            executable='online_ipp.py',
            name='online_ipp'
        ),

        Node(
            package='ros_sgp_tools',
            executable='path_planner.py',
            name='path_planner'
        )
    ])
