#date: 2025-07-10T17:15:37Z
#url: https://api.github.com/gists/0e3c0325f6700b124e8cc7ec22a627d2
#owner: https://api.github.com/users/shima-nct

#!/usr/bin/env python3
"""
MechArm 270 Pi：各軸動作チェック
--------------------------------
全 6 軸を順番に +30° → -30° → 0° へ動かし、
エンコーダー値を表示して確認するスクリプト。

使い方：
  python mecharm_axis_check.py  [/dev/ttyACM0]  [1000000]

Ctrl-C でいつでも安全に原点 (0,0,0,0,0,0) へ戻して終了。
"""

import sys
import time
from pymycobot import MechArm270, PI_PORT, PI_BAUD

# ---- 引数処理 --------------------------------------------------------------
port = sys.argv[1] if len(sys.argv) > 1 else PI_PORT
baudrate = int(sys.argv[2]) if len(sys.argv) > 2 else PI_BAUD

# ---- 接続 -----------------------------------------------------------------
arm = MechArm270(port, baudrate)
print(f"[INFO] Connected: {port} @ {baudrate} bps")
time.sleep(0.5)  # UART 安定待ち

# ---- 動作チェック ----------------------------------------------------------
HOME = [0, 0, 0, 0, 0, 0]
STEP = 30  # ±30°


def move_wait(target, speed=60):
    arm.send_angles(target, speed)
    while arm.is_moving() == 1:
        time.sleep(0.05)
    if arm.is_moving() == -1:
        raise Exception("Move failed")

try:
    arm.power_on()
    time.sleep(0.5)
    if arm.is_power_on() == -1:
        raise Exception("Power on failed")

    # 原点へ移動
    print("[STEP] Home position")
    move_wait(HOME)

    for idx in range(6):
        print(f"\n[STEP] Axis {idx+1} +{STEP}°")
        plus = HOME.copy()
        plus[idx] = STEP
        minus = HOME.copy()
        minus[idx] = -STEP

        move_wait(plus)
        print("  Angles:", arm.get_angles())

        print(f"[STEP] Axis {idx+1} -{STEP}°")
        move_wait(minus)
        print("  Angles:", arm.get_angles())

        print("[STEP] Return to 0°")
        move_wait(HOME)
        print("  Angles:", arm.get_angles())

    print("\n[SUCCESS] All axis test finished.")

except KeyboardInterrupt:
    print("\n[INTERRUPT] Returning to home and exiting…")
    move_wait(HOME)

finally:
    arm.power_off()
    print("[EXIT] Servos released. Bye!")
