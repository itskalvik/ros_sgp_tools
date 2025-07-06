import os
import sys
import psutil

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


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
    data_type = get_var('DATA_TYPE' ,'GazeboPing1D')
    data_folder = get_var('DATA_FOLDER', '')
    fcu_url = get_var('FCU_URL', 'udp://0.0.0.0:14550@')
    ping1d_port = get_var('PING1D_PORT', '/dev/ttyUSB0')

    geofence_plan = PathJoinSubstitution([FindPackageShare('ros_sgp_tools'),
                                          'launch', 'data',
                                          'mission.plan'])
    config_file = PathJoinSubstitution([FindPackageShare('ros_sgp_tools'),
                                        'launch', 'data',
                                        'config.yaml'])

    nodes = []
    path_planner = Node(package='ros_sgp_tools',
                        executable='path_planner.py',
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

    if data_type=='Ping1D':
        # Ping1D ROS package 
        sensor = Node(package='ping_sonar_ros',
                      executable='ping1d_node',
                      name='Ping1D',
                      parameters=[
                        {'port': ping1d_port}
                      ])
        nodes.append(sensor)   
    elif data_type=='GazeboPing1D':
        # Gazebo ROS Bridge
        bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[f'ping1d@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'],
        )
        nodes.append(bridge)         

    return LaunchDescription(nodes)