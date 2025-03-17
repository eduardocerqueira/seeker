#date: 2025-03-17T17:04:17Z
#url: https://api.github.com/gists/9cc3e47886d8df1f8bc5976ff354a01a
#owner: https://api.github.com/users/TheSpace-hub

import matplotlib.pyplot as plt
import numpy as np

# Класс для реализации системы непересекающихся множеств (DSU)
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            return True
        return False

# Функция для вычисления расстояния между двумя точками
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Алгоритм Крускала для нахождения минимального остовного дерева
def kruskal_mst(points):
    n = len(points)
    edges = []

    # Создаем список всех рёбер
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((distance(points[i], points[j]), i, j))

    # Сортируем рёбра по весу
    edges.sort()

    # Инициализируем DSU
    dsu = DSU(n)
    mst = []

    # Добавляем рёбра в MST, если они не создают цикл
    for weight, u, v in edges:
        if dsu.union(u, v):
            mst.append((u, v, weight))

    return mst

# Функция для визуализации MST
def plot_mst(points, mst):
    plt.scatter(points[:, 0], points[:, 1], color='red')  # Рисуем точки
    for u, v, weight in mst:
        plt.plot([points[u][0], points[v][0]], [points[u][1], points[v][1]], color='blue')  # Рисуем рёбра MST
    plt.title("Минимальное остовное дерево")
    plt.show()

# Создаем случайные точки
points = np.random.rand(10, 2)

# Находим минимальное остовное дерево
mst = kruskal_mst(points)

# Визуализируем результат
plot_mst(points, mst)
