from sensor_msgs.msg import NavSatFix, Range, FluidPressure, Image
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from message_filters import Subscriber
from cv_bridge import CvBridge
import numpy as np


class SensorCallback:
    def __init__(self):
        pass

    def get_subscriber(self):
        pass
    
    def process_msg(self, msg):
        pass

class GPSData(SensorCallback):
    def get_subscriber(self, node_obj):
        sub =  Subscriber(node_obj, NavSatFix, 
                          "mavros/global_position/global",
                          qos_profile=qos_profile_sensor_data)
        return sub
    
    def process_msg(self, msg):
        return np.array([msg.latitude, msg.longitude, msg.altitude])
    

class SonarData(SensorCallback):
    def get_subscriber(self, node_obj):
        sub =  Subscriber(node_obj, Range, 
                          "mavros/rangefinder_pub",
                          qos_profile=qos_profile_sensor_data)
        return sub
    
    def process_msg(self, msg, position):
        return [position[:2]], [msg.range]


class PressureData(SensorCallback):
    def __init__(self):
        self.data_mean = None

    def get_subscriber(self, node_obj):
        sub =  Subscriber(node_obj, FluidPressure, 
                          "mavros/imu/static_pressure",
                          qos_profile=qos_profile_sensor_data)
        return sub
    
    def process_msg(self, msg, position):
        if self.data_mean is None:
            self.data_mean = msg.fluid_pressure
        return [position[:2]], [msg.fluid_pressure-self.data_mean]


class ZEDData(SensorCallback):
    def __init__(self):
        # Setup variables to get data from depth map
        self.bridge = CvBridge()

        # Get data from 3x3 grid
        delta = 1280//6
        mask_x = [delta, delta*3, delta*5]
        delta = 720//6
        mask_y = [delta, delta*3, delta*5]
        mask_x, mask_y = np.meshgrid(mask_x, mask_y)
        self.mask_x = mask_x.reshape(-1)
        self.mask_y = mask_y.reshape(-1)        
        self.bool_mask = np.array([[-1,  1], [0,  1], [1,  1],
                                   [-1,  0], [0,  0], [1,  0],
                                   [-1, -1], [0, -1], [1, -1]])

        # ZED FoV 110° (H) x 70° (V)
        # Mapped to right triangle base for 1 meter height
        self.fov_scale = np.array([1.42815, 0.70021])
        self.dist_scale = 1/111111 # 1 meter offset in lat/long

    def get_subscriber(self, node_obj):
        sub =  Subscriber(node_obj, Image, 
                          "zed/zed_node/depth/depth_registered",
                          qos_profile=QoSProfile(depth=10))
        return sub
    
    def process_msg(self, msg, position):
        data_X, data_y = [], []
        # scale FoV spread in proportion to the height from the ground
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        height = np.mean(depth[np.where(np.isfinite(depth))])
        for i, point in enumerate(zip(self.mask_x, self.mask_y)):
            data = depth[point[1], point[0]]
            if np.isfinite(data):
                data_X.append(position[:2] + \
                              (self.bool_mask[i]*height*self.fov_scale*self.dist_scale))
                data_y.append(data)

        return data_X, data_y