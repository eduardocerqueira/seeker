#date: 2025-06-27T16:38:41Z
#url: https://api.github.com/gists/34a56e6831d529da9363cc169504182f
#owner: https://api.github.com/users/eviarch

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math

matplotlib.use('TKAgg')  # 强制使用 TkAgg 后端

plt.rc("font", family='Microsoft YaHei')  # family是设置的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

P3 = np.array([2, 0])
P2 = np.array([2, 3])
P1 = np.array([-2, 3])
P0 = np.array([-2, 6])
bezier_weight = 0.2

# 存储点
x_vals = []
t_vals = []
b3x_vals = []
b3y_vals = []
slope_vals = []
arctan_vals = []
cp_vals = []
merge_target_vals = []


def bezier_slope(t):
    dx_dt = 3 * (1 - t) ** 2 * (P1[0] - P0[0]) + 6 * (1 - t) * t * (P2[0] - P1[0]) + 3 * t ** 2 * (P3[0] - P2[0])
    dy_dt = 3 * (1 - t) ** 2 * (P1[1] - P0[1]) + 6 * (1 - t) * t * (P2[1] - P1[1]) + 3 * t ** 2 * (P3[1] - P2[1])

    if dx_dt == 0:
        return float('inf')  # 垂直切线
    return dy_dt / dx_dt


def control_point_angle(x, y, control_point):
    dy = control_point[1] - y
    dx = control_point[0] - x
    if dx == 0:
        return float('inf')
    return math.degrees(math.atan(dy/dx))


def euclidean_distance(point1_x, point1_y, point2_x, point2_y):
    return abs(math.sqrt((point2_x - point1_x)**2 + (point2_y - point1_y)**2))


for y in np.arange(0, 6.01, 0.01):  # 包含6，防止浮点误
    delt_y = 6

    t = (math.pow((((-1/4) + (y/(2*delt_y))) + math.pow((((-1/4) + (y/(2*delt_y)))**2 + (1/4)**3), 1/2)), 1/3) -
         math.pow(-(((-1/4) + (y/(2*delt_y))) - math.pow((((-1/4) + (y/(2*delt_y)))**2 + (1/4)**3), 1/2)), 1/3) + 0.5)
    print(f"Calculate t: {t}")

    B3_x = (1 - t) ** 3 * P0[0] + 3 * (1 - t) ** 2 * t * P1[0] + 3 * (1 - t) * t ** 2 * P2[0] + t ** 3 * P3[0]
    B3_y = (1-t)**3 * P0[1] + 3*(1-t)**2 * t * P1[1] + 3*(1-t)*t**2 * P2[1] + t**3 * P3[1]
    slope = bezier_slope(t)
    arctan = math.degrees(math.atan(1.0/slope))
    cp_angle = (control_point_angle(B3_x, B3_y, P3))
    #                 math.sqrt(euclidean_distance(B3_x, B3_y, P3[0], P3[1]) / euclidean_distance(P0[0], P0[1], P3[0], P3[1]))
    cp_weight = (math.sqrt(euclidean_distance(B3_x, B3_y, P3[0], P3[1]) / euclidean_distance(P0[0], P0[1], P3[0], P3[1])))

    # merge_target_yaw = (arctan * bezier_weight + cp_angle * (1 - bezier_weight))
    merge_target_yaw = arctan * (1 - cp_weight) + cp_angle * cp_weight

    print(f"B3_x: {B3_x}")
    print(f"B3_y: {B3_y}")
    print(f"slope: {slope}")
    print(f"arctan: {arctan}")
    print(f"control_point_angle: {cp_angle}\n")
    x_vals.append(y)  # y 是横坐标
    t_vals.append(t)
    b3x_vals.append(B3_x)
    b3y_vals.append(B3_y)
    slope_vals.append(slope)
    arctan_vals.append(arctan)
    cp_vals.append(cp_angle)
    merge_target_vals.append(merge_target_yaw)


# 图像绘制
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(x_vals, t_vals, s=5, color='blue')
plt.title("t vs. x (where x = y input)")
plt.xlabel("x")
plt.ylabel("t")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(b3x_vals, x_vals, s=5, color='red')
plt.title("B3_y vs. B3_x ")
plt.xlabel("B3_x")
plt.ylabel("B3_y")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(x_vals, b3y_vals, s=5, color='green')
plt.title("B3_y vs. x (Bezier y value)")
plt.xlabel("x")
plt.ylabel("B3_y")
plt.grid(True)

# plt.figure(figsize=(15, 5))  # 设置图像大小，可根据需要调整
# plt.subplot(1, 3, 1)
# plt.scatter(x_vals, slope_vals, s=5, color='blue')
# plt.title("slope vs. x (Bezier curve)")
# plt.xlabel("x")
# plt.ylabel("slope")
# plt.ylim(-50, 0)  # 设置 y 轴范围
# plt.grid(True)

# plt.subplot(1, 3, 2)
# plt.scatter(x_vals, arctan_vals, s=5, color='red')
# plt.title("arctan vs. x (Bezier curve)")
# plt.xlabel("x")
# plt.ylabel("arctan")
# plt.ylim(-60, 0)  # 设置 y 轴范围
# plt.grid(True)

# plt.subplot(1, 3, 3)
# plt.scatter(x_vals, cp_vals, s=5, color='green')
# plt.title("cp_angles vs. x (Bezier curve)")
# plt.xlabel("x")
# plt.ylabel("arctan")
# plt.grid(True)

# plt.figure(figsize=(5, 5))

# plt.subplot(1, 1, 1)
# plt.scatter(x_vals, merge_target_vals, s=5, color='blue')
# plt.title("merge_target_yaw vs. x")
# plt.xlabel("x")
# plt.ylabel("t")
# plt.grid(True)

# plt.tight_layout()
# plt.show()
