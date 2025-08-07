import os
import sys
import yaml
import psutil

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def get_var(var, default):
    try:
        return os.environ[var]
    except:
        return default

def generate_launch_description():
    # Sanity check to avoid crashing the system because of low memory
    total_memory = psutil.virtual_memory().total + psutil.swap_memory().total
    if total_memory < 4e9:
        sys.exit("Low memory (less than 4GB), increase swap size!")

    # Get parameter values
    geofence_plan = os.path.join(get_package_share_directory('ros_sgp_tools'), 
                                                             'launch', 'data',
                                                             'mission.plan')
    config_file = os.path.join(get_package_share_directory('ros_sgp_tools'), 
                                                           'launch', 'data',
                                                           'config.yaml')
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    sensor = config['robot']['sensor']
    data_folder = get_var('DATA_FOLDER', '')
    fcu_url = get_var('FCU_URL', 'udp://0.0.0.0:14550@')

    nodes = []
    path_planner = Node(package='ros_sgp_tools',
                        executable='path_planner.py',
                        output='screen',
                        parameters=[
                            {'geofence_plan': geofence_plan,
                             'config_file': config_file,
                             'data_folder': data_folder
                            }
                        ])
    nodes.append(path_planner)

    # MAVROS controller
    path_follower = Node(package='ros_sgp_tools',
                         executable='path_follower.py',
                         parameters=[{'xy_tolerance': 1.0}])
    nodes.append(path_follower)

    # MAVROS
    mavros = GroupAction(
                    actions=[
                        IncludeLaunchDescription(
                            XMLLaunchDescriptionSource([
                                PathJoinSubstitution([
                                    FindPackageShare('mavros_control'),
                                    'launch',
                                    'mavros.launch'
                                ])
                            ]),
                            launch_arguments={
                                "fcu_url": fcu_url
                            }.items()
                        ),
                    ]
                )
    nodes.append(mavros)

    if sensor=='Ping1D':
        # Ping1D ROS package 
        ping1d_port = config['sensor'][sensor]['port']
        sensor = Node(package='bluerobotics_sonar',
                       executable='ping1d',
                       parameters=[{
                            'mode_auto': 1,
                            'port': ping1d_port
                       }],
                       output='screen')
        nodes.append(sensor)
    elif sensor=='GazeboPing1D':
        # Gazebo ROS Bridge
        bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[f'ping1d@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'],
        )
        nodes.append(bridge)         

    return LaunchDescription(nodes)