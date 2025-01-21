#date: 2025-01-21T17:02:54Z
#url: https://api.github.com/gists/7e50f5116bac7317020d5f50000c9d04
#owner: https://api.github.com/users/codezakh

from enum import Enum
from typing import (
    Generic,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
)
import random

from pydantic import BaseModel, Field
from typing_extensions import Self
from ulid import ULID

T = TypeVar("T")

class Node(BaseModel):
    """Class defining the interface for a node in the MCTS search tree."""
    ulid: ULID = Field(default_factory=ULID)
    parent: Optional[Self] = Field(default=None, repr=False)
    children: list[Self] = Field(default_factory=list, repr=False)
    visits: int = 0
    total_value: float = 0 

    @property
    def is_leaf(self) -> bool:
        return len(self.children) > 0
    
    @property
    def best_child(self) -> Self:
        return max(self.children, key=lambda n: n.ucb_score)

    @property
    def ucb_score(self) -> float:
        """Calculate the UCB1 score for this node."""
        if self.visits == 0:
            return float('inf')

        if self.parent is None:
            return self.total_value / self.visits
        
        exploitation = self.total_value / self.visits
        exploration = (2 * (self.parent.visits + 1)) ** 0.5 / (self.visits + 1)
        return exploitation + exploration

    def update_stats(self, simulation_value: float) -> None:
        """Update node statistics after a simulation."""
        self.visits += 1
        self.total_value += simulation_value

    def get_lineage(self) -> list[Self]:
        """
        Get the lineage of nodes from the root to the current node.
        """
        lineage: list[Self] = []
        current = self
        while current:
            lineage.append(current)
            current = current.parent
        return lineage

class SuccessorFunction(Protocol):
    """Function that returns the successors of a given node."""

    def __call__(self, state: Node) -> Sequence[Node]: ...


class IsTerminalTestFunction(Protocol):
    """Function that returns whether a given node is a terminal node."""

    def __call__(self, state: Node) -> bool: ...


class StateEvaluationFunction(Protocol):
    """Function that returns a score for a given node."""

    def __call__(self, state: Node) -> float: ...


class MctsResult(BaseModel):
    trajectories: list[list[Node]]
    best_next_node: Node


class MonteCarloTreeSearch:
    def __init__(
        self,
        initial_state: Node,
        successor_fn: SuccessorFunction,
        check_is_terminal_fn: IsTerminalTestFunction,
        evaluation_fn: StateEvaluationFunction,
        max_rollout_depth: int = 100,
    ):
        self.root = initial_state
        self.successor_fn = successor_fn
        self.check_is_terminal_node = check_is_terminal_fn
        self.evaluation_fn = evaluation_fn
        self.max_rollout_depth = max_rollout_depth

    def select(self, node: Node) -> Node:
        """
        Select the most promising child node to visit.
        """
        current = node
        while not current.is_leaf:
            current = current.best_child
        return current
    
    def expand(self, node: Node) -> Node:
        """
        Expand the node by sampling child nodes from the successor function.
        """
        if not node.children:
            successors = self.successor_fn(node)
            node.children.extend(successors)
        return random.choice(node.children)
    

    def simulate(self, node: Node) -> tuple[float, list[Node]]:
        """
        Perform a rollout from the current node to a terminal node.
        """
        current = node
        depth = 0
        trajectory: list[Node] = []

        # Rollout until we reach a terminal node or hit the rollout length limit. 
        while True:
            trajectory.append(current)

            if depth >= self.max_rollout_depth:
                break

            if self.check_is_terminal_node(current):
                break

            successors = self.successor_fn(current)
            
            if not successors:
                break

            current = successors[0]
            depth += 1

        return self.evaluation_fn(current), trajectory

    def backpropagate(self, node: Node, value: float) -> None:
        """
        Backpropagate the value of the simulation to the root node.
        This updates the statistics of all nodes in the lineage.
        """
        current = node
        while current:
            current.update_stats(value)
            current = current.parent

    
    def search(self, search_root: Node, rollouts: int) -> MctsResult:
        """
        Perform MCTS from a given root node to select the best next step.
        """
        trajectories: list[list[Node]] = []
        for _ in range(rollouts):
            # Select the most promising child node to visit.
            node = self.select(search_root)

            # If the node is not a terminal node, expand it.
            if not self.check_is_terminal_node(node):
                node = self.expand(node)

            # Simulate a rollout from the node to a terminal node
            # to estimate the value of the node.
            value, trajectory = self.simulate(node)
            trajectories.append(trajectory)

            # Backpropagate the value to the root node.
            self.backpropagate(node, value)
        return MctsResult(trajectories=trajectories, best_next_node=self.root.best_child)