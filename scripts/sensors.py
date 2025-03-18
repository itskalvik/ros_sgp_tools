from sensor_msgs.msg import NavSatFix, Range, FluidPressure, Image, LaserScan
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

class GPS(SensorCallback):
    def get_subscriber(self, node_obj, callback_group=None):
        sub =  Subscriber(node_obj, NavSatFix, 
                          "mavros/global_position/global",
                          qos_profile=qos_profile_sensor_data,
                          callback_group=callback_group)
        return sub
    
    def process_msg(self, msg):
        return np.array([msg.latitude, msg.longitude, msg.altitude])

class SerialPing1D(SensorCallback):
    def __init__(self):
        self.topic = "mavros/rangefinder_pub"

    def get_subscriber(self, node_obj, callback_group=None):
        sub = Subscriber(node_obj, Range, 
                         self.topic,
                         qos_profile=qos_profile_sensor_data,
                         callback_group=callback_group)
        return sub
    
    def process_msg(self, msg, position):
        return [position[:2]], [msg.range]
    
class GazeboPing1D(SensorCallback):
    def __init__(self):
        self.topic = "Ping1D"

    def get_subscriber(self, node_obj, callback_group=None):
        sub = Subscriber(node_obj, LaserScan, 
                         self.topic,
                         qos_profile=qos_profile_sensor_data,
                         callback_group=callback_group)
        return sub
    
    def process_msg(self, msg, position):
        return [position[:2]], [np.mean(msg.ranges)]

class Ping1D(SerialPing1D):
    def __init__(self):
        self.topic = "ping1d/range"

class Pressure(SensorCallback):
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

class ZED(SensorCallback):
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
        # 110°*(4/6) x 70°*(4/6) FoV adjusted for 3x3 grid locations
        # = 73.33° x 46.67°
        # Mapped to right triangle base for 1 meter height and angle set to FoV
        self.fov_scale = np.array([0.74443, 0.43139])
        self.dist_scale = 1/111111 # 1 meter offset distance in lat/long

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

        data_X = np.array(data_X).astype(float)
        data_y = np.array(data_y).astype(float)

        return data_X, data_y