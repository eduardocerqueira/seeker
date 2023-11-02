#date: 2023-11-02T17:01:11Z
#url: https://api.github.com/gists/1a97fec793f92667bda387ffb62e6426
#owner: https://api.github.com/users/snowbellows

import time
import psutil
from pprint import pprint
INTERFACE = "wlp3s0"

CHANNEL_LIST = list(range(1,12))

def chan_to_freq(channel):
    return 2412+(channel-1)*5

def run_bash_command(cmd: str, allow_err=False, suppress_log=False):
    import subprocess

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, error = process.communicate()
    if error and not allow_err:
        print(f"Got error {error.decode()}")
        raise Exception(error)
    else:
        return output


def calc_throughput_kbps(sample_time):
    sample_a = psutil.net_io_counters(pernic=True)
    time.sleep(sample_time)
    sample_b = psutil.net_io_counters(pernic=True)
    
    throughput_kbps = (sample_b[INTERFACE].bytes_recv-sample_a[INTERFACE].bytes_recv)/1_000/sample_time
    return throughput_kbps

def set_wpa_channel(channel):
    run_bash_command(f"sudo wpa_cli chan_switch 5 {chan_to_freq(channel)} -i {INTERFACE}", allow_err=True,)

if __name__ == "__main__":
    data = []
    for channel in CHANNEL_LIST:
        print(f"Checking {channel}")
        set_wpa_channel(channel)
        data.append([channel,round(calc_throughput_kbps(5),2)])
    data.sort(key=lambda x: x[1],reverse=True)
    print(f"Check complete. Best channel is {data[0][0]} with throughput of {data[0][1]}kbps")
    print("Full data below:")
    pprint(data)