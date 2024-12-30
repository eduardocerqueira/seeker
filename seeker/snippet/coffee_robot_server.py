#date: 2024-12-30T16:49:04Z
#url: https://api.github.com/gists/5c7eab9ebe27fbb28ea1a58b290a6f0c
#owner: https://api.github.com/users/sulevsky

import time

import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node

from make_coffee_interfaces.action import MakeCoffee


class CoffeeRobotServerNode(Node):
    def __init__(self):
        super().__init__("coffee_robot_server")
        self.action_server = ActionServer(
            self,
            action_type=MakeCoffee,
            action_name="make_coffee",
            goal_callback=self.goal_callback,
            execute_callback=self.execute_callback,
        )
        self.get_logger().info("coffee_robot_server has been initalized")

    def goal_callback(self, goal: MakeCoffee.Goal) -> GoalResponse:
        logger = self.get_logger()
        ordered_coffee_cups = goal.ordered_coffee_cups
        self.get_logger().info(f"Received a goal: {goal=}")
        if ordered_coffee_cups <= 0:
            logger.warn(f"Goal is rejected, {goal=}, not valid amount of coffee cups")
            return GoalResponse.REJECT
        elif ordered_coffee_cups > 1000:
            logger.warn(f"Goal is rejected, {goal=}, too many coffee cups requested")
            return GoalResponse.REJECT
        else:
            logger.info(f"Goal is accepted, {goal=}")
            return GoalResponse.ACCEPT

    def execute_callback(self, goal_handle: ServerGoalHandle) -> MakeCoffee.Result:
        self.get_logger().info("execute callback is called")
        # get data from the make_coffee_interfaces.action.MakeCoffee.Goal
        goal = goal_handle.request
        self.get_logger().info(f"goal is: {goal}")
        ordered_coffee_cups = goal.ordered_coffee_cups
        self.get_logger().info(f"ordered {ordered_coffee_cups} coffee cups")

        # main executon logic
        prepared_coffee_cups = self.prepare_coffee(ordered_coffee_cups)

        # updating goal state
        goal_handle.succeed()

        # returning result (i.e. make_coffee_interfaces.action.MakeCoffee.Result)
        result = MakeCoffee.Result()
        result.prepared_coffee_cups_total = prepared_coffee_cups
        return result

    def prepare_coffee(self, ordered_coffee_cups: int) -> int:
        # simple function to highlight some work is done
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
