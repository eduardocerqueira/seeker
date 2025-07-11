#date: 2025-07-11T17:01:04Z
#url: https://api.github.com/gists/5e348eac75c59344cd5de3c431ebd200
#owner: https://api.github.com/users/maxstrid

import math
import numpy as np

class TrapezoidMotionProfile:
    def __init__(self, distance, max_vel, max_accel) -> None:
        self.ta = max_vel / max_accel
        accel_distance = 0.5 * max_accel * self.ta**2

        if (accel_distance > distance / 2.0):
            self.ta = math.sqrt(distance / max_accel)

        accel_distance = 0.5 * max_accel * self.ta**2

        self.tc = distance - 2 * accel_distance

        if (self.tc < 0):
            self.tc = 0

        self.max_accel = max_accel

    def accel(self, t) -> float:
        if t < self.ta:
            return self.max_accel
        elif t > (self.tc + self.ta) and t < (2 * self.ta + self.tc):
            return -self.max_accel
        else:
            return 0

    def position(self, t) -> float:
        if t <= self.ta:
            return 0.5 * self.max_accel * t**2
        elif t <= (self.tc + self.ta):
            d1 = 0.5 * self.max_accel * self.ta**2
            d2 = self.max_accel * self.ta * (t - self.ta)
            return d1 + d2
        else:
            d1 = 0.5 * self.max_accel * self.ta**2
            d2 = self.max_accel * self.ta * self.tc
            d3 = 0.5 * -self.max_accel * (t - self.ta - self.tc)**2 + self.max_accel * self.ta * (t - self.ta - self.tc)
            return d1 + d2 + d3

    def total_time(self):
        return 2 * self.ta + self.tc

class SystemPlant:
    def __init__(self, J, Kt, B, max_current, process_noise_std_dev, sensor_noise_std_dev) -> None:
        self.J = J
        self.Kt = Kt
        self.B = B
        self.process_noise_std_dev = process_noise_std_dev
        self.sensor_noise_std_dev = sensor_noise_std_dev
        self.max_current = max_current
        self.pos = 0.0
        self.vel = 0.0

    def apply_current(self, current, dt) -> None:
        if (abs(current) >= self.max_current):
            current = np.sign(current) * self.max_current

        torque = current * self.Kt - (self.B * self.vel)

        accel = torque / self.J + np.random.normal(loc=0.0, scale=self.process_noise_std_dev)

        self.vel += accel * dt
        self.pos += self.vel * dt

    def get_position(self) -> float:
        return self.pos + np.random.normal(loc=0.0, scale=self.sensor_noise_std_dev)


def main():
    trapezoidal_profile = TrapezoidMotionProfile(1.0, 0.5, 0.2)

    plant = SystemPlant(
        1.0, # J in kgm^2
        880, # Kt in Nm/A
        0.0, # B
        50, # Max Current
        0.1, # Process Noise Std Dev
        0.1, # Sensor noise std dev
    )

    last_error = 0
    integral = 0
    dt = 0.005
    for t in np.arange(0, trapezoidal_profile.total_time() + 5.0 + dt, dt):
        accel_goal = trapezoidal_profile.accel(t)
        position_goal = trapezoidal_profile.position(t)
        current_goal = (plant.J * accel_goal) / plant.Kt

        position = plant.get_position()

        Kp = 0.5
        Ki = 0.0
        Kd = 0.0

        error = (position_goal - position)

        derror = (error - last_error) / dt
        last_error = error

        integral += error

        current_output = current_goal + Kp * error + Ki * integral + Kd * derror

        plant.apply_current(current_output, dt)