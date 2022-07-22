#date: 2022-07-22T17:02:11Z
#url: https://api.github.com/gists/18c0829de579133a6246213f6b242686
#owner: https://api.github.com/users/FabienDanieau

#   Copyright POLLEN ROBOTICS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory import InterpolationMode
import time
import cv2 as cv
import numpy as np

from threading import Thread

thread_running = False

def look_at(target):
    reachy.head.look_at(
        target[0], target[1], target[2], 1.0)


def move_arm(target, duration):
    # Inverse kinematics. See https://docs.pollen-robotics.com/sdk/first-moves/kinematics/
    target_matrix = np.array([
        [0, 0, -1, target[0]],
        [0, 1, 0, target[1]],
        [1, 0, 0, target[2]],
        [0, 0, 0, 1],
    ])

    joint_pos_ball = reachy.r_arm.inverse_kinematics(target_matrix)
    reachy.turn_on('r_arm')

    # use the goto function
    goto({joint: pos for joint, pos in zip(
         reachy.r_arm.joints.values(), joint_pos_ball)}, duration=duration)


def throw_movement():
    # Forward kinematics
    reachy.r_arm.r_gripper.goal_position = -60
    right_angled_position = {
        reachy.r_arm.r_shoulder_pitch: -90,
        reachy.r_arm.r_shoulder_roll: 0,
        reachy.r_arm.r_arm_yaw: 0,
        reachy.r_arm.r_elbow_pitch: 0,
        reachy.r_arm.r_forearm_yaw: 0,
        reachy.r_arm.r_wrist_pitch: 0,
        reachy.r_arm.r_wrist_roll: 10,
    }
    goto(
        goal_positions=right_angled_position,
        duration=0.2,
        interpolation_mode=InterpolationMode.MINIMUM_JERK
    )

def basketball():
    # Ball position in Reachy coordinate system. provided by Unity script
    ball_position = np.array([0.3907774,-0.2557656,-0.372798])
    look_at(ball_position)
    #move reachy arm above the ball
    avoid_ball = np.array([ball_position[0]-0.05, ball_position[1]-0.10, 0])
    move_arm(avoid_ball, 1.0)
    above_ball = np.array([ball_position[0]-0.05, ball_position[1], 0])
    move_arm(above_ball, 1.0)
    #open the gripper
    reachy.r_arm.r_gripper.goal_position = -50
    time.sleep(1.0)
    #reach the ball
    move_arm(ball_position, 4.0)
    time.sleep(2.0)
    #close gripper
    reachy.r_arm.r_gripper.goal_position = -40
    time.sleep(0.5)
    # move arm up and throw the ball
    move_arm(above_ball, 4.0)
    throw_movement()
    time.sleep(1)
    global thread_running
    thread_running = False


if __name__ == "__main__":
    # replace with correct IP if the simulation is not on your computer
    reachy = ReachySDK(host='localhost')
    
    t = Thread(target=lambda: basketball())
    t.daemon = True
    t.start()
    
    thread_running = True

    while (thread_running):
        cv.imshow('Right camera', reachy.right_camera.last_frame)
        cv.waitKey(30)
    
    cv.destroyAllWindows()
    
    reachy.turn_off_smoothly('r_arm')

    exit()