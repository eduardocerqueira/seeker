#date: 2023-04-12T16:59:28Z
#url: https://api.github.com/gists/60f0958c490c6b0ed902d09e9b988243
#owner: https://api.github.com/users/sloretz

#!/usr/bin/env python3
# topics:      /zed2i/zed_node/left/camera_info               : sensor_msgs/CameraInfo 
#              /zed2i/zed_node/left/image_rect_color          : sensor_msgs/Image      
#              /zed2i/zed_node/point_cloud/cloud_registered   : sensor_msgs/PointCloud2
#              /zed2i/zed_node/right/camera_info              : sensor_msgs/CameraInfo 
#              /zed2i/zed_node/right/image_rect_color         : sensor_msgs/Image

import math
import time
import threading

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

import rospy

def sec_to_stamp(time_sec, stamp):
    stamp.secs = int(time_sec)
    stamp.nsecs = int((time_sec - int(time_sec)) * 1e9)


def fake_left_camera_info(time_sec):
    msg = CameraInfo()
    sec_to_stamp(time_sec, msg.header.stamp)
    msg.header.frame_id = "zed2i_left_camera_optical_frame"
    msg.height = 540
    msg.width = 960
    msg.distortion_model = "plumb_bob"
    msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.K = [555.6596069335938, 0.0, 490.7663269042969, 0.0, 555.6596069335938, 255.26710510253906, 0.0, 0.0, 1.0]
    msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.P = [555.6596069335938, 0.0, 490.7663269042969, 0.0, 0.0, 555.6596069335938, 255.26710510253906, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.binning_x = 0
    msg.binning_y = 0
    msg.roi.x_offset = 0
    msg.roi.y_offset = 0
    msg.roi.height = 0
    msg.roi.width = 0
    msg.roi.do_rectify = False
    return msg


def fake_right_camera_info(time_sec):
    msg = CameraInfo()
    sec_to_stamp(time_sec, msg.header.stamp)
    msg.header.frame_id = "zed2i_right_camera_optical_frame"
    msg.height = 540
    msg.width = 960
    msg.distortion_model = "plumb_bob"
    msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.K = [555.6596069335938, 0.0, 490.7663269042969, 0.0, 555.6596069335938, 255.26710510253906, 0.0, 0.0, 1.0]
    msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.P = [555.6596069335938, 0.0, 490.7663269042969, -66.45494842529297, 0.0, 555.6596069335938, 255.26710510253906, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.binning_x = 0
    msg.binning_y = 0
    msg.roi.x_offset = 0
    msg.roi.y_offset = 0
    msg.roi.height = 0
    msg.roi.width = 0
    msg.roi.do_rectify = False
    return msg


def fake_left_image(time_sec):
    msg = fake_left_image.msg
    if msg is None:
        fake_left_image.msg = Image()
        msg = fake_left_image.msg
        msg.header.frame_id = "zed2i_left_camera_optical_frame"
        msg.height = 540
        msg.width = 960
        msg.encoding = "bgra8"
        msg.is_bigendian = 0
        msg.step = 3840
        msg.data = bytes([d % 255 for d in range(2073600)])
    sec_to_stamp(time_sec, msg.header.stamp)
    return msg

fake_left_image.msg = None


def fake_right_image(time_sec):
    msg = fake_right_image.msg
    if msg is None:
        fake_right_image.msg = Image()
        msg = fake_right_image.msg
        msg.header.frame_id = "zed2i_right_camera_optical_frame"
        msg.height = 540
        msg.width = 960
        msg.encoding = "bgra8"
        msg.is_bigendian = 0
        msg.step = 3840
        msg.data = bytes([d % 255 for d in range(2073600)])
    sec_to_stamp(time_sec, msg.header.stamp)
    return msg

fake_right_image.msg = None


def fake_points(time_sec):
    msg = fake_points.msg
    if msg is None:
        fake_points.msg = PointCloud2()
        msg = fake_points.msg
        msg.header.frame_id = "zed2i_left_camera_frame"
        msg.height = 540
        msg.width = 960
        msg.fields = []
        x_field = PointField()
        x_field.name = "x"
        x_field.offset = 0
        x_field.datatype = 7
        x_field.count = 1
        msg.fields.append(x_field)
        y_field = PointField()
        y_field.name = "y"
        y_field.offset = 4
        y_field.datatype = 7
        y_field.count = 1
        msg.fields.append(y_field)
        z_field = PointField()
        z_field.name = "z"
        z_field.offset = 8
        z_field.datatype = 7
        z_field.count = 1
        msg.fields.append(z_field)
        rgb_field = PointField()
        rgb_field.name = "rgb"
        rgb_field.offset = 12
        rgb_field.datatype = 7
        rgb_field.count = 1
        msg.fields.append(rgb_field)
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 15360
        msg.data = bytes([d % 255 for d in range(8294400)])
        msg.is_dense = False
    sec_to_stamp(time_sec, msg.header.stamp)
    return msg

fake_points.msg = None


def loop_30hz(left_camera_info_pub, right_camera_info_pub):
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        now = time.time()
        left_camera_info_pub.publish(fake_left_camera_info(now))
        right_camera_info_pub.publish(fake_right_camera_info(now))
        rate.sleep()


def loop_15hz(left_image_pub, right_image_pub):
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        now = time.time()
        left_image_pub.publish(fake_left_image(now))
        right_image_pub.publish(fake_right_image(now))
        rate.sleep()


def loop_10hz(points_pub):
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        now = time.time()
        points_pub.publish(fake_points(now))
        rate.sleep()


def main():
    rospy.init_node('fake_zed2i')
    # 30 Hz
    left_camera_info_pub = rospy.Publisher('/zed2i/zed_node/left/camera_info', CameraInfo, queue_size=1)
    right_camera_info_pub = rospy.Publisher('/zed2i/zed_node/right/camera_info', CameraInfo, queue_size=1)
    # 15 Hz
    left_image_pub = rospy.Publisher('/zed2i/zed_node/left/image_rect_color', Image, queue_size=1)
    right_image_pub = rospy.Publisher('/zed2i/zed_node/right/image_rect_color', Image, queue_size=1)
    # 10 Hz
    points_pub = rospy.Publisher('/zed2i/zed_node/point_cloud/cloud_registered', PointCloud2, queue_size=1)

    thread_30hz = threading.Thread(target=loop_30hz, args=(left_camera_info_pub, right_camera_info_pub))
    thread_15hz = threading.Thread(target=loop_15hz, args=(left_image_pub, right_image_pub))
    thread_10hz = threading.Thread(target=loop_10hz, args=(points_pub,))

    thread_30hz.start()
    thread_15hz.start()
    thread_10hz.start()
    thread_30hz.join()
    thread_15hz.join()
    thread_10hz.join()


if __name__ == '__main__':
    main()