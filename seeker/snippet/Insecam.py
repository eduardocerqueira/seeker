#date: 2023-12-15T16:42:17Z
#url: https://api.github.com/gists/bbd4f89c235d6f91f70195c932fe9c5a
#owner: https://api.github.com/users/rafa-br34

import latest_user_agents
import contextlib
import threading
import requests
import socket
import urllib
import time

from urllib3.exceptions import InsecureRequestWarning
from urllib.parse import urlparse
from colorama import Fore, Back, Style
import bs4 as bs


c_Headers = {
	"Accept": "image/avif,image/webp,*/*",
	"Accept-Language": "pt-BR,pt;q=0.8,en-US;q=0.5,en;q=0.3",
	"Accept-Encoding": "gzip, deflate, br",
	"Connection": "keep-alive",
	"Referer": "https://www.google.com/",
	"Sec-Fetch-Dest": "image",
	"Sec-Fetch-Mode": "no-cors",
	"Sec-Fetch-Site": "same-site",
	"DNT": "1",
	"Sec-GPC": "1"
}
c_Threads = 80
c_Timeout = 3
c_Ports = [554, 23]
c_SSL = False


class Host:
	Address = ""
	Ports = []

	def __init__(self, Address, Port):
		self.Address = Address
		self.Port = isinstance(Port, list) and [int(P) for P in Port] or [int(Port)]

	def CheckPort(self, Port, Timeout=10):
		with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as Socket:
			Socket.settimeout(Timeout)
			if Socket.connect_ex((self.Address, Port)) == 0:
				return True
		return False

	def CheckPorts(self, Ports, Timeout=10):
		Checks = []
		for Port in Ports:
			Checks.append(self.CheckPort(Port, Timeout))
		return Checks


class _Shared:
	HostCache = []
	Threads = []
	Hosts = []
	Run = True


g_Shared = _Shared()


def WorkerLogic():
	while g_Shared.Run:
		if len(g_Shared.Hosts) <= 0:
			time.sleep(0.01)
			continue
		Target = g_Shared.Hosts.pop()

		if Target.Address in g_Shared.HostCache: continue
		if len(g_Shared.HostCache) > 2000: g_Shared.HostCache.pop(0)
		g_Shared.HostCache.append(Target.Address)

		Result = Target.CheckPorts(c_Ports, c_Timeout)

		Refined = list(filter(None, [Result[i] and p or None for i, p in enumerate(c_Ports)]))
		if len(Refined) > 0:
			Complete = len(Refined) == len(c_Ports)
			String = f"{Target.Address} {str(Refined)}"
			
			if Complete:
				print(Fore.GREEN + Target.Address + Style.RESET_ALL)
			else:
				#print(Fore.RED + String + Style.RESET_ALL)
				pass


def main():
	for _ in range(c_Threads):
		Thread = threading.Thread(target=WorkerLogic)
		Thread.start()
		g_Shared.Threads.append(Thread)

	Page = 1

	Session = requests.Session()

	if c_SSL == False:
		requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
	try:
		while g_Shared.Run:
			if len(g_Shared.Hosts) > 5000:
				time.sleep(1)
				continue
			NewHeaders = c_Headers.copy()
			NewHeaders["User-Agent"] = latest_user_agents.get_random_user_agent()
			Result = Session.get(f"http://www.insecam.org/en/byrating/?page={Page}", headers=NewHeaders, verify=c_SSL)
			if Result.status_code != 200:
				time.sleep(5)
				continue

			Soup = bs.BeautifulSoup(Result.text, "html.parser")
			Cameras = Soup.find_all("img", "thumbnail-item__img img-responsive")

			for Camera in Cameras:
				URL = urlparse(Camera.get("src")).netloc
				g_Shared.Hosts.append(Host(*URL.split(':')))
				

			if len(Cameras) > 0:
				Page += 1
			else:
				break
	finally:
		g_Shared.Run = False

	for Thread in g_Shared.Threads:
		Thread.join()
	
	

if __name__ == '__main__':
	main()