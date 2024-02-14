#!/usr/bin/env python3

import os
import sys
import rospy
from ros_sgp_ipp.msg import RSSI


def get_rssi():
    cmd = os.popen('nmcli dev wifi | grep "^*"').read()
    cmd = cmd.split()
    if len(cmd) > 6:
        dBm = float(cmd[7])/2-100
    else:
        rospy.logerr("Cannot get info from wireless!")
        dBm = -1.
    return dBm

def get_pi_rssi():
    cmd = os.popen('iwconfig wlan0 | grep Quality').read()
    cmd = cmd.split()
    if len(cmd) > 4:
        dBm = float(cmd[3].split('=')[1])
    else:
        rospy.logerr("Cannot get info from wireless!")
        dBm = -1.
    return dBm

def main(rssi_fun):
    rospy.init_node('rssi_publisher', anonymous=True)
    publisher = rospy.Publisher('rssi', RSSI, queue_size=10)
    rate = rospy.Rate(10)
    rospy.loginfo('RSSI publisher initialized, publishing RSSI')

    while not rospy.is_shutdown():
        dBm = rssi_fun()
        msg = RSSI()
        msg.rssi = dBm
        msg.header.stamp = rospy.Time.now()
        publisher.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        PI = sys.argv[1]
    else:
        PI = False

    if PI:
        rssi_fun = get_pi_rssi
    else:
        rssi_fun = get_rssi

    try:
        main(rssi_fun)
    except rospy.ROSInterruptException:
        pass