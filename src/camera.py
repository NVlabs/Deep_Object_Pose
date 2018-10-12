#!/usr/bin/env python
# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
This file opens an RGB camera and publishes images via ROS. 
It uses OpenCV to capture from camera 0.
"""

from __future__ import print_function
import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ImageSensor

from sensor_msgs.msg import Image as Image_msg
import numpy as np
import cv2

# Global variables
cam_index = 0  # index of camera to capture
topic = '/dope/webcam_rgb_raw'  # topic for publishing
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print("ERROR:  Unable to open camera for capture.  Is camera plugged in?")
    exit(1)
    
def publish_images(freq=5):
    rospy.init_node('dope_webcam_rgb_raw', anonymous=True)
    images_out = rospy.Publisher(topic, Image_msg, queue_size=10)
    rate = rospy.Rate(freq)

    print ("Publishing images from camera {} to topic '{}'...".format(
            cam_index, 
            topic
            )
    )
    print ("Ctrl-C to stop")
    while not rospy.is_shutdown():
        ret, frame = cap.read()

        if ret:
            msg_frame_edges = CvBridge().cv2_to_imgmsg(frame, "bgr8")
            images_out.publish(msg_frame_edges)

        rate.sleep()

if __name__ == "__main__":
    
    try :
        publish_images()
    except rospy.ROSInterruptException:
        pass

