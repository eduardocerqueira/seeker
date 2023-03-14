#date: 2023-03-14T17:11:34Z
#url: https://api.github.com/gists/fc311ebfda03e9453e569d31a4aab1a2
#owner: https://api.github.com/users/jaybrecht

#!/usr/bin/env python3

from math import pi

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.duration import Duration

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class RobotControl(Node):
    floor_robot_joint_names = ['floor_shoulder_pan_joint',
                               'floor_shoulder_lift_joint',
                               'floor_elbow_joint',
                               'floor_wrist_1_joint',
                               'floor_wrist_2_joint',
                               'floor_wrist_3_joint',]

    floor_robot_home_joint_positions = [0.0, -pi/2, pi/2, -pi/2, -pi/2, 0.0]

    ceiling_robot_joint_names = ['ceiling_shoulder_pan_joint',
                                 'ceiling_shoulder_lift_joint',
                                 'ceiling_elbow_joint',
                                 'ceiling_wrist_1_joint',
                                 'ceiling_wrist_2_joint',
                                 'ceiling_wrist_3_joint',]

    ceiling_robot_home_joint_positions = [0.0, -pi/2, pi/2, pi, -pi/2, 0.0]

    def __init__(self):
        super().__init__('robot_control')

        sim_time = Parameter(
            "use_sim_time",
            Parameter.Type.BOOL,
            True
        )

        self.set_parameters([sim_time])

        self._floor_robot_action_client = ActionClient(
            self, FollowJointTrajectory, '/floor_robot_controller/follow_joint_trajectory')

        self._ceiling_robot_action_client = ActionClient(
            self, FollowJointTrajectory, '/ceiling_robot_controller/follow_joint_trajectory')
        
        self.floor_robot_at_home = False
        self.ceiling_robot_at_home = False

    def move_floor_robot_home(self, move_time):
        point = JointTrajectoryPoint()
        point.positions = self.floor_robot_home_joint_positions
        point.time_from_start = Duration(seconds=move_time).to_msg()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.floor_robot_joint_names
        goal_msg.trajectory.points.append(point)

        self._floor_robot_action_client.wait_for_server()

        self._floor_robot_send_goal_future = self._floor_robot_action_client.send_goal_async(
            goal_msg)

        self._floor_robot_send_goal_future.add_done_callback(
            self.floor_robot_goal_response_callback)

    def move_ceiling_robot_home(self, move_time):
        point = JointTrajectoryPoint()
        point.positions = self.ceiling_robot_home_joint_positions
        point.time_from_start = Duration(seconds=move_time).to_msg()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.ceiling_robot_joint_names
        goal_msg.trajectory.points.append(point)

        self._ceiling_robot_action_client.wait_for_server()

        self._ceiling_robot_send_goal_future = self._ceiling_robot_action_client.send_goal_async(
            goal_msg)

        self._ceiling_robot_send_goal_future.add_done_callback(
            self.ceiling_robot_goal_response_callback)

    def floor_robot_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._floor_robot_get_result_future = goal_handle.get_result_async()
        self._floor_robot_get_result_future.add_done_callback(
            self.floor_robot_get_result_callback)
    
    def ceiling_robot_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._ceiling_robot_get_result_future = goal_handle.get_result_async()
        self._ceiling_robot_get_result_future.add_done_callback(
            self.ceiling_robot_get_result_callback)

    def floor_robot_get_result_callback(self, future):
        result = future.result().result
        result: FollowJointTrajectory.Result

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info("Move succeeded")
        else:
            self.get_logger().error(result.error_string)

        self.floor_robot_at_home = True
    
    def ceiling_robot_get_result_callback(self, future):
        result = future.result().result
        result: FollowJointTrajectory.Result

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info("Move succeeded")
        else:
            self.get_logger().error(result.error_string)

        self.ceiling_robot_at_home = True
