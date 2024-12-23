#date: 2024-12-23T16:36:33Z
#url: https://api.github.com/gists/dcb521655b3bcbe8888410b15beb3c7f
#owner: https://api.github.com/users/sulevsky

import time

import rclpy
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node

from make_coffee_interfaces.action import MakeCoffee


class CoffeeRobotServerNode(Node):
    def __init__(self):
        super().__init__("coffee_robot_server")
        self.action_server = ActionServer(
            node=self,
            action_type=MakeCoffee,
            action_name="make_coffee",
            execute_callback=self.execute_callback,
        )
        self.get_logger().info("coffee_robot_server has been initalized")

    def execute_callback(self, goal_handle: ServerGoalHandle) -> MakeCoffee.Result:
        self.get_logger().info("execute callback is called")
        # get data from the Goal
        goal = goal_handle.request
        self.get_logger().info(f"goal is: {goal}")
        ordered_coffee_cups = goal.ordered_coffee_cups
        self.get_logger().info(f"ordered {ordered_coffee_cups} coffee cups")

        # main executon logic
        prepared_coffee_cups = self.prepare_coffee(ordered_coffee_cups)

        # updating goal state
        goal_handle.succeed()

        # returning result
        result = MakeCoffee.Result()
        result.prepared_coffee_cups_total = prepared_coffee_cups
        return result

    def prepare_coffee(self, ordered_coffee_cups: int) -> int:
        for i in range(ordered_coffee_cups):
            time.sleep(1)
            self.get_logger().info(f"prepared one coffee cup, index: {i}")
        return ordered_coffee_cups


def main(args=None):
    rclpy.init(args=args)
    node = CoffeeRobotServerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
