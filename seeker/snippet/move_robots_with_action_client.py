#date: 2023-03-14T17:12:14Z
#url: https://api.github.com/gists/7fc0b07666497873d9ccb80416f00393
#owner: https://api.github.com/users/jaybrecht

#!/usr/bin/env python3
'''
Example script to move the robot using ROS2 Actions

To test this script, run the following command:

- ros2 launch ariac_gazebo ariac.launch.py trial_name:=tutorial
- ros2 run ariac_tutorials move_robot_with_action_client.py
'''

import rclpy
from rclpy.executors import MultiThreadedExecutor
from ariac_tutorials.competition_interface import CompetitionInterface
from ariac_tutorials.robot_control import RobotControl

def main(args=None):
    '''
    main function for the move_robot_with_action_client script.

    Args:
        args (Any, optional): ROS arguments. Defaults to None.
    '''
    rclpy.init(args=args)

    interface = CompetitionInterface()
    robot_control = RobotControl()

    executor = MultiThreadedExecutor()

    executor.add_node(interface)
    executor.add_node(robot_control)

    interface.start_competition()
    interface.wait(3) # Wait for controllers to come online

    robot_control.move_floor_robot_home(3.0)
    robot_control.move_ceiling_robot_home(5.0)
    
    while rclpy.ok():
        try:
            executor.spin_once()
            if robot_control.floor_robot_at_home and robot_control.ceiling_robot_at_home:
                break
        except KeyboardInterrupt:
            break

    rclpy.shutdown()

if __name__ == '__main__':
    main()
