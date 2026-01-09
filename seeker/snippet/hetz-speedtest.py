#date: 2026-01-09T17:10:18Z
#url: https://api.github.com/gists/0ed2c38d9439bc9ceb9a6dcdb89b9f02
#owner: https://api.github.com/users/delacosta456

import sys
import time
import requests
import math
import subprocess
import platform

hosts = {
  "fsn1-speed.hetzner.com": {
    "sm": "https://fsn1-speed.hetzner.com/100MB.bin",
    "md": "https://fsn1-speed.hetzner.com/1GB.bin",
    "lg": "https://fsn1-speed.hetzner.com/10GB.bin",
  },
  "hel1-speed.hetzner.com": {
    "sm": "https://hel1-speed.hetzner.com/100MB.bin",
    "md": "https://hel1-speed.hetzner.com/1GB.bin",
    "lg": "https://hel1-speed.hetzner.com/10GB.bin",
  },
  "speed.hetzner.de": {
    "sm": "https://speed.hetzner.de/100MB.bin",
    "md": "https://speed.hetzner.de/1GB.bin",
    "lg": "https://speed.hetzner.de/10GB.bin",
  },
  "ash.icmp.hetzner.com": {
    "sm": "http://ash.icmp.hetzner.com/100MB.bin",
    "md": "http://ash.icmp.hetzner.com/1GB.bin",
    "lg": "http://ash.icmp.hetzner.com/10GB.bin",
  },
  "hil.icmp.hetzner.com": {
    "sm": "http://hil.icmp.hetzner.com/100MB.bin",
    "md": "http://hil.icmp.hetzner.com/1GB.bin",
    "lg": "http://hil.icmp.hetzner.com/10GB.bin",
  }
}

def downloadFile(url):
  points = []
  with open("/dev/null", 'wb') as f:
    start = time.process_time()
    r = requests.get(url, stream=True)
    total_length = int(r.headers.get('content-length'))
    dl = 0
    if total_length is None: # no content length header
      f.write(r.content)
    else:
      for chunk in r.iter_content(1024):
        dl += len(chunk)
        f.write(chunk)
        point = dl//(time.process_time() - start)
        points.append(point)
  avg = round(sum(points)/len(points), 2)
  return avg

# get latency of host
def get_latency(host):
  if platform.system() == "Windows":
    return 0
  try:
    output = subprocess.check_output(["ping", "-c", "1", host])
    output = output.decode("utf-8")
    output = output.split("\n")
    for line in output:
      if "time=" in line:
        return float(line.split("time=")[1].split(" ")[0])
  except:
    return 0

def convert_size(size_bytes: int) -> str:
  if size_bytes == 0:
    return "0B"
  size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
  i = int(math.floor(math.log(size_bytes, 1024)))
  p = math.pow(1024, i)
  s = round(size_bytes / p, 2)
  return f"{s}{size_name[i]}"

# convert size to bits per second
def convert_speed(size_bytes: int) -> int:
  return size_bytes * 8

def main() :
  size = sys.argv[1]
  for host in hosts:
    file = hosts[host][size]
    print(f"Downloading {file} from {host}")
    (avg_speed) = downloadFile(file)
    print(f"Average speed: {convert_size(avg_speed)}/s")
    if platform.system() == "Windows":
      print("Can't measure latency on Windows")
      continue
    print(f"Latency: {get_latency(host)}ms")

if __name__ == "__main__" :
  main()
