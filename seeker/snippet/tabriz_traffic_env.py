#date: 2025-08-25T17:08:20Z
#url: https://api.github.com/gists/1010989d4e976fe26d1a5d4b2c533cff
#owner: https://api.github.com/users/ayhanasghari

# tabriz_traffic_env.py

import gym
from gym import spaces
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

class TabrizTrafficEnv(gym.Env):
    def __init__(self, graph_path="tabriz_iran.graphml"):
        super().__init__()

        # بارگذاری گراف خیابانی تبریز از فایل محلی
        self.graph = ox.load_graphml(graph_path)
        self.intersections = list(self.graph.nodes)

        # تعریف فضای مشاهدات و اکشن‌ها
        self.num_signals = len(self.intersections)
        self.observation_space = spaces.Box(low=0, high=20, shape=(self.num_signals,), dtype=np.int32)
        self.action_space = spaces.MultiBinary(self.num_signals)  # 0: حفظ فاز، 1: تغییر فاز

        self.reset()

    def reset(self):
        # صف خودروها در هر تقاطع (ساده‌شده)
        self.queues = np.random.randint(0, 10, size=self.num_signals)
        # وضعیت چراغ‌ها: 0 یا 1
        self.lights = np.zeros(self.num_signals, dtype=np.int32)
        return self.queues.copy()

    def step(self, action):
        assert self.action_space.contains(action), "❌ Invalid action shape"

        # تغییر فاز چراغ‌ها
        self.lights = (self.lights + action) % 2

        # عبور خودروها در جهت سبز
        passed = np.where(self.lights == 1, np.minimum(self.queues, 3), 0)
        self.queues -= passed

        # ورود خودروهای جدید
        arrivals = np.random.randint(0, 3, size=self.num_signals)
        self.queues += arrivals

        # محاسبه پاداش: منفی مجموع صف‌ها
        reward = -np.sum(self.queues)

        # پایان اپیزود؟ نه فعلاً
        done = False

        return self.queues.copy(), reward, done, {}

    def render(self, mode="human"):
        print("🚦 Traffic Queues:")
        for i, q in enumerate(self.queues):
            print(f"  Node {self.intersections[i]} → Queue: {q} | Light: {'Green' if self.lights[i] else 'Red'}")


   
        fig, ax = plt.subplots(figsize=(10, 10))

        # موقعیت گره‌ها
        node_colors = ["green" if light else "red" for light in self.lights]
        node_sizes = [300 + 50*q for q in self.queues]  # اندازه بر اساس صف

        # رسم گراف با رنگ چراغ‌ها و اندازه صف‌ها
        ox.plot_graph(self.graph, ax=ax, node_color=node_colors, node_size=node_sizes,
                    show=False, close=False)

        plt.title("🚦 وضعیت چراغ‌ها و صف خودروها در تقاطع‌های تبریز")
        plt.show()
