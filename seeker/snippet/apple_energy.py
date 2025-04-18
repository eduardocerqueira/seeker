#date: 2025-04-18T17:07:59Z
#url: https://api.github.com/gists/6c94b66910299dfe6c3cb863576b9cdc
#owner: https://api.github.com/users/michahn01

from dataclasses import asdict
from zeus_apple_silicon import AppleEnergyMonitor, AppleEnergyMetrics

def dummy_work(limit):
    x = 0
    for i in range(limit):
        for j in range(10000):
            x += i + j
    return x

def print_result(res):
    res_dict = {
        "cpu_total_mj": res.cpu_total_mj,
        "efficiency_cores_mj": res.efficiency_cores_mj,
        "performance_cores_mj": res.performance_cores_mj,
        "efficiency_core_manager_mj": res.efficiency_core_manager_mj,
        "performance_core_manager_mj": res.performance_core_manager_mj,
        "dram_mj": res.dram_mj,
        "gpu_mj": res.gpu_mj,
        "gpu_sram_mj": res.gpu_sram_mj,
        "ane_mj": res.ane_mj,
    }
    parts = []
    for k, v in res_dict.items():
        parts.append(f"{k:30}: {v}")
    print("\n".join(parts))


mon = AppleEnergyMonitor()

mon.begin_window("window")
dummy_work(1000)
res = mon.end_window("window")

print_result(res)
