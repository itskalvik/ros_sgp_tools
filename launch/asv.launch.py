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
    mission_type = config['robot']['mission_type']
    data_folder = get_var('DATA_FOLDER', '')
    fcu_url = get_var('FCU_URL', 'udp://0.0.0.0:14550@')

    nodes = []
    # Path Planner
    if mission_type=='Waypoint':
        executable = 'waypoint_planner.py'
    elif mission_type=='IPP':
        executable = 'ipp_planner.py'
    elif mission_type=='AdaptiveIPP':
        executable = 'adaptive_ipp_planner.py'
    elif mission_type=='Coverage':
        executable = 'coverage_planner.py'

    path_planner = Node(package='ros_sgp_tools',
                        executable=executable,
                        parameters=[
                            {'geofence_plan': geofence_plan,
                             'config_file': config_file,
                             'data_folder': data_folder
                            }],
                        output='screen')
    nodes.append(path_planner)

    # MAVROS Controller
    path_follower = Node(package='ros_sgp_tools',
                         executable='path_follower.py',
                         parameters=[{'xy_tolerance': 0.7,
                                      'geofence_plan': geofence_plan,
                                     }],
                         arguments=['--controller', 'mavros'],
                         output='screen')
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

    # Sensor Node
    if sensor=='Ping1D':
        # Ping1D ROS package 
        sensor = Node(package='bluerobotics_sonar',
                      executable='ping1d',
                      parameters=[config['sensor'][sensor]],
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