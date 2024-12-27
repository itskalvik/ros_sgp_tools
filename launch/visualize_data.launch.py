from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.actions import GroupAction, IncludeLaunchDescription, DeclareLaunchArgument


def generate_launch_description():
    nodes = []

    # Mission log folder
    mision_log_arg = DeclareLaunchArgument('mission_log', default_value='',
                                           description='Mission log folder')
    nodes.append(mision_log_arg)

    # Number of samples
    num_samples_arg = DeclareLaunchArgument('num_samples', default_value='5000',
                                            description='Number of samples used to visualize the data')
    nodes.append(num_samples_arg)

    # Point cloud publisher
    pcd_publisher = Node(package='ros_sgp_tools',
                         executable='data_visualizer.py',
                         name='pcd_publisher',
                         parameters=[
                            {'mission_log': LaunchConfiguration('mission_log'),
                             'num_samples': LaunchConfiguration('num_samples')}
                         ])
    nodes.append(pcd_publisher)

    # Foxglove (web-based rviz)
    foxglove = GroupAction(
                    actions=[
                        IncludeLaunchDescription(
                            XMLLaunchDescriptionSource([
                                PathJoinSubstitution([
                                    FindPackageShare('foxglove_bridge'),
                                    'launch',
                                    'foxglove_bridge_launch.xml'
                                ])
                            ]),
                        )
                    ]
                )
    nodes.append(foxglove)

    return LaunchDescription(nodes)