#!/usr/bin/env python3

import os
import rospy
from ros_sgp_ipp.msg import RSSI


def loop():
    rospy.init_node('rssi_publisher', anonymous=True)
    publisher = rospy.Publisher('rssi', RSSI, queue_size=10)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
    
        myCmd = os.popen('nmcli dev wifi | grep "^*"').read()
        cmdList = myCmd.split()
        
        if len(cmdList) > 6:
            quality = float(cmdList[7])
            dBm = quality/2-100
            msg = RSSI()
            msg.rssi = dBm
            msg.header.stamp = rospy.Time.now()
            publisher.publish(msg)
        else:
            rospy.logerr("Cannot get info from wireless!")
        rate.sleep()

if __name__ == '__main__':
    try:
        loop()
    except rospy.ROSInterruptException:
        pass