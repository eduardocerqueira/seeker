#date: 2025-08-25T17:08:20Z
#url: https://api.github.com/gists/1010989d4e976fe26d1a5d4b2c533cff
#owner: https://api.github.com/users/ayhanasghari

# agents/agent.py

import networkx as nx
import random

class Agent:
    def __init__(self, agent_id, graph, start_node=None, goal_node=None):
        self.id = agent_id
        self.graph = graph
        self.start_node = start_node or random.choice(list(graph.nodes))
        self.goal_node = goal_node or random.choice(list(graph.nodes))
        self.current_node = self.start_node
        self.path = []
        self.memory = []
        self.reward = 0

    def plan_path(self):
        try:
            self.path = nx.shortest_path(self.graph, source=self.current_node, target=self.goal_node)
        except nx.NetworkXNoPath:
            self.path = []

    def move(self):
        if self.path and len(self.path) > 1:
            self.current_node = self.path.pop(1)
            self.memory.append(self.current_node)
        else:
            self.plan_path()

    def reset(self):
        self.current_node = self.start_node
        self.path = []
        self.memory = []
        self.reward = 0
