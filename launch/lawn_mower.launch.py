from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    geofence_plan = PathJoinSubstitution([FindPackageShare('ros_sgp_tools'),
                                          'launch', 'data',
                                          'lawn_mover_mission.plan'])

    return LaunchDescription([Node(package='ros_sgp_tools',
                                   executable='aqua_controller.py',
                                   name='IPPMission',
                                   parameters=[
                                       {'geofence_plan': geofence_plan}
                                       ])
                            ])