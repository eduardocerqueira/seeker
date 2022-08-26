#date: 2022-08-26T16:59:41Z
#url: https://api.github.com/gists/b4ea46de2c7a3d138287307cba61a7ae
#owner: https://api.github.com/users/versatran01

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:35:23 2022

@author: chao
"""

import numpy as np
from scipy.spatial.transform import Rotation

import rospy

from geometry_msgs.msg import Point as PointMsg
from geometry_msgs.msg import Quaternion as QuaternionMsg
from geometry_msgs.msg import PoseStamped as PoseStampedMsg
from nav_msgs.msg import Path as PathMsg


def rostime_from_sec(sec: float) -> rospy.rostime.Time:
    return rospy.rostime.Time.from_sec(sec)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit quaternion"""
    assert q.ndim == 1 and q.size == 4, q.shape
    return q / np.linalg.norm(q)


def quat_msg_from_array(q: np.ndarray) -> QuaternionMsg:
    msg = QuaternionMsg()
    msg.x, msg.y, msg.z, msg.w = normalize_quat(q.squeeze())
    return msg


def point_msg_from_array(p: np.ndarray) -> PointMsg:
    msg = PointMsg()
    msg.x, msg.y, msg.z = p.squeeze()
    return msg


def normalize_tum(data: np.ndarray) -> np.ndarray:
    """Normalize tum data such that first pose is identity"""
    ts_w_i = data[:, 1:4]
    Rs_w_i = Rotation.from_quat(data[:, 4:])

    R_0_w = Rs_w_i[0].inv()
    Rs_0_i = R_0_w * Rs_w_i
    ts_0_i = R_0_w.apply(ts_w_i - ts_w_i[0])

    data_norm = np.empty_like(data)
    data_norm[:, 0] = data[:, 0]
    data_norm[:, 1:4] = ts_0_i
    data_norm[:, 4:] = Rs_0_i.as_quat()

    return data_norm


def path_msg_from_tum(data: np.ndarray) -> PathMsg:
    time = data[:, 0]
    pos = data[:, 1:4]
    quat = data[:, 4:]

    path_msg = PathMsg()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp = rostime_from_sec(time[0])

    for i in range(data.shape[0]):
        pose_msg = PoseStampedMsg()
        pose_msg.header.frame_id = frame_id
        pose_msg.header.stamp = rostime_from_sec(time[i])

        pose_msg.pose.position = point_msg_from_array(pos[i])
        pose_msg.pose.orientation = quat_msg_from_array(quat[i])

        path_msg.poses.append(pose_msg)

    return path_msg


if __name__ == "__main__":
    rospy.init_node("tum_traj_pub")
    frame_id = "world"
    pub_rate = 1

    frame_id = rospy.get_param("~frame", frame_id)
    pub_rate = rospy.get_param("~rate", pub_rate)
    tum_file = rospy.get_param("~file")

    print(f"frame_id: {frame_id}")
    print(f"pub_rate: {pub_rate}")
    print(f"Reading tum file: {tum_file}")

    data = np.loadtxt(tum_file, delimiter=" ")
    data = normalize_tum(data)
    path_msg = path_msg_from_tum(data)
    print(f"Total poses in gt file: {len(path_msg.poses)}")

    traj_pub = rospy.Publisher("gt", PathMsg, queue_size=10)

    rosrate = rospy.Rate(pub_rate)

    while not rospy.is_shutdown():
        traj_pub.publish(path_msg)
        rosrate.sleep()