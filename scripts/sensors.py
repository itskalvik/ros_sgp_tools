from sensor_msgs.msg import NavSatFix, Range, FluidPressure
from rclpy.qos import qos_profile_sensor_data
from message_filters import Subscriber


class SensorCallback:
    def __init__(self):
        pass

    def get_subscriber(self):
        pass
    
    def process_msg(self, msg, **kwargs):
        pass


class SonarData(SensorCallback):
    def get_subscriber(self, node_obj):
        sub =  Subscriber(node_obj, Range, 
                          "mavros/rangefinder_pub",
                          qos_profile=qos_profile_sensor_data)
        return sub
    
    def process_msg(self, msg, **kwargs):
        return msg.range


class GPSData(SensorCallback):
    def get_subscriber(self, node_obj):
        sub =  Subscriber(node_obj, NavSatFix, 
                          "mavros/global_position/global",
                          qos_profile=qos_profile_sensor_data)
        return sub
    
    def process_msg(self, msg, **kwargs):
        return msg.latitude, msg.longitude, msg.altitude
    

class PressureData(SensorCallback):
    def __init__(self):
        self.data_mean = None

    def get_subscriber(self, node_obj):
        sub =  Subscriber(node_obj, FluidPressure, 
                          "mavros/imu/static_pressure",
                          qos_profile=qos_profile_sensor_data)
        return sub
    
    def process_msg(self, msg, **kwargs):
        if self.data_mean is None:
            self.data_mean = msg.fluid_pressure
        return msg.fluid_pressure-self.data_mean