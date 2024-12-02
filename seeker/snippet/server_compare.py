#date: 2024-12-02T17:00:26Z
#url: https://api.github.com/gists/62003ca810543ec9d57a8dcecbe1d789
#owner: https://api.github.com/users/PandeCode

#!/bin/env python3
import os
import re

from fire import Fire
import json5 as json
import matplotlib.pyplot as plt
from numpy import Infinity
from typing import Iterable

defaultServers: set[str] = {
	"1.1.1.1",  # Cloudflare
	"1.0.0.1",  # Cloudflare
	"208.67.222.222",  # OpenDNS
	"208.67.220.220",  # OpenDNS
	"8.8.4.4",  # Google Public DNS ipv4
	"8.8.8.8",  # Google Public DNS ipv4
	"9.9.9.11",  # Quad9 ECS enabled
	"149.112.112.11",  # Quad9 ECS enabled
	"9.9.9.10",  # Quad9 Unsecured: No Malware blocking, no DNSSEC validation
	"149.112.112.10",  # Quad9 Unsecured: No Malware blocking, no DNSSEC validation
	"9.9.9.9",  # Quad9
	"149.112.112.112",  # Quad9
	"185.228.168.9",  # CleanBrowsing
	"185.228.169.9",  # CleanBrowsing
	"185.228.168.168",  # CleanBrowsing
	"76.76.19.19",  # Alternate DNS
	"76.223.122.150",  # Alternate DNS
	"94.140.14.14",  # AdGuard DNS
	"94.140.15.15",  # AdGuard DNS
	"176.103.130.130",  # AdGuard DNS
	"91.239.100.100",  # UncensoredDNS
	"89.233.43.71",  # UncensoredDNS
	"84.200.69.80",  # DNS.WATCH
	"84.200.70.40",  # DNS.WATCH
	"8.26.56.26",  # ComodoSecureDNS
	"8.20.247.20",  # ComodoSecureDNS
	"77.88.8.8",  # Yandex.DNS
	"77.88.8.7",  # Yandex DNS
	"77.88.8.1",  # Yandex.DNS
	"13.239.157.177",  # OpenNIC
	"172.98.193.42",  # OpenNIC
	"66.70.228.164",  # OpenNIC
	"205.171.3.66",  # CenturyLink(Level3)
	"205.171.202.166",  # CenturyLink(Level3)
	"195.46.39.39",  # SafeDNS
	"195.46.39.40",  # SafeDNS
	"198.101.242.72",  # Alternate DNS
	"64.6.65.6",  # Verisign Public DNS
	"74.82.42.42",  # HurricaneElectric
	"109.69.8.51",  # puntCAT
	"216.146.35.35",  # Dyn
	"216.146.36.36",  # Dyn
	"45.33.97.5",  # FreeDNS
	"37.235.1.177",  # FreeDNS
	"64.6.64.6",  # Neustar
	"64.6.65.6",  # Neustar
	"45.77.165.194",  # FourthEstate
	"45.32.36.36",  # FourthEstate
}
defaultServers = { "filemoon.sx","streamtape.com","www.mp4upload.com","mcloud.to","vidstream.pro" }

def main(
	old: bool = False,
	iterations: int = 10,
	servers: Iterable[str] | str = defaultServers,
	flash: bool = False,
) -> None: 
	if type(servers) == str:
		servers = servers.split(",")
	
	output: dict[str, str] = dict()

	if old:
		try:
			with open(r"./output.json", "r") as f:
				output = json.load(f)  # type: ignore
		except Exception as e:
			print("Error: ", e)
			output = getData(iterations, writeToFile=True, servers=servers, flash=flash)

	else:
		output = getData(iterations, writeToFile=True, servers=servers, flash=flash)
	showData(output)


def getData(
	iterations: int,
	servers: Iterable[str] = defaultServers,
	writeToFile: bool = False,
	flash: bool = False,
) -> dict[str, str]:

	output = dict()
	for server in servers:
		command = f"ping -A -c {iterations} -i {'0' if flash else '0.2'} {server}"
		print(command)
		output[server] = os.popen( command).read()

	if writeToFile:
		with open("output.json", "w") as file:
			json.dump(output, file)

	return output


def showData(output: dict[str, str]):

	maxPing = -Infinity
	minPing = Infinity

	maxPingHolder = ""
	minPingHolder = ""

	maxAveragePing = -Infinity
	minAveragePing = Infinity

	maxAveragePingHolder = ""
	minAveragePingHolder = ""

	averages: dict[str, float] = {}
	mins: dict[str, float] = {}
	maxes: dict[str, float] = {}

	for key, value in output.items():
		if not value:
			print(key)
		data = value.split()
		pings: list[float] = []

		for i in data:
			if i.startswith("time="):
				pings.append(float(i[5:]))

		byNl = value.split("\n")
		maxAvgMinLine = byNl[len(byNl) - 2]

		if not maxAvgMinLine:
			continue
		thisMax, thisAvg, thisMin, *_ = list(
			map(float, re.findall(r"(\d+\.\d+)", maxAvgMinLine))
		)

		averages[key] = thisAvg
		maxes[key] = thisMax
		mins[key] = thisMin

		if thisMax > maxPing:
			maxPing = thisMax
			maxPingHolder = key
		if thisMin < minPing:
			minPing = thisMin
			minPingHolder = key

		if thisAvg > maxAveragePing:
			maxAveragePing = thisAvg
			maxAveragePingHolder = key
		if thisAvg < minAveragePing:
			minAveragePing = thisMin
			minAveragePingHolder = key

		print(f"{key} {thisAvg} {thisMax} {thisMin}")

		plt.plot(
			[i for i in range(1, len(pings) + 1)], pings, label=key, marker="o",
		)

	print(f"Max ping:         {maxPing} by {maxPingHolder}")
	print(f"Min ping:         {minPing} by {minPingHolder}")
	print(f"Max Average ping: {maxAveragePing} by {maxAveragePingHolder}")
	print(f"Min Average ping: {minAveragePing} by {minAveragePingHolder}")

	plt.xlabel("Run")
	plt.ylabel("Ping")
	plt.title("Ping against runs")
	plt.grid(True)
	plt.legend()

	plt.show()


if __name__ == "__main__":
	Fire(main)
