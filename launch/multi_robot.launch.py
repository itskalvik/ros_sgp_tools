from launch_ros.actions import Node
from launch import LaunchDescription
from launch_ros.actions import PushRosNamespace
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def generate_launch_description():
    namespace = 'robot_0'
    
    return LaunchDescription([
        Node(
            package='ros_sgp_tools',
            executable='online_ipp.py',
            namespace=namespace,
            name='online_ipp'
        ),

        Node(
            package='ros_sgp_tools',
            executable='ipp_mission.py',
            namespace=namespace,
            name='ipp_mission'
        ),

        GroupAction(
            actions=[
                # push_ros_namespace to set namespace of included nodes
                PushRosNamespace(namespace),
                IncludeLaunchDescription(
                    XMLLaunchDescriptionSource([
                        PathJoinSubstitution([
                            FindPackageShare('ros_sgp_tools'),
                            'launch',
                            'mavros.launch'
                        ])
                    ]),
                    launch_arguments={
                    }.items()
                )
            ]
        )
    ])
