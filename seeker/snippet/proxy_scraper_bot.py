#date: 2022-03-17T17:11:01Z
#url: https://api.github.com/gists/e5a451e604ffc8c05a98e179b0435804
#owner: https://api.github.com/users/pvww

"""
Usage:
    Step 1:
        Setup libraries with following commands:
        python3 -m pip install -U pip setuptools wheel
        python3 -m pip install -U pyrogram tgcrypto aiocron requests beautifulsoup4
    
    Step 2:
        Edit lines 75, 76, 78, 80 and 87
    
    Step 3:
        Run the file :)))
"""

from datetime import datetime
from io import BytesIO

from aiocron import crontab
from bs4 import BeautifulSoup
from pyrogram import Client, idle
from requests import get, session

proxies = []


def hidemy_name():
    url = "https://hidemy.name/en/proxy-list/?start={}#list"
    ses = session()
    ses.headers = {
        "user-agent": "Mozilla/5.0 (X11; CrOS aarch64 0.54.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3936.2 Safari/537.36"
    }
    for _ in range(0, 400, 50):
        req = ses.get(url.format(_))
        soup = BeautifulSoup(req.text, "html.parser")
        for tr in soup.find("tbody").find_all("tr"):
            ip, port, *_ = tr.find_all("td")
            proxies.append(f"{ip.text}:{port.text}")


def proxyscrape():
    req = get(
        "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks4&timeout=10000&country=all")
    for proxy in req.text.split():
        proxies.append(proxy)


def us_proxy():
    req = get("https://www.us-proxy.org")
    soup = BeautifulSoup(req.text, "html.parser")
    for tr in soup.find("tbody").find_all("tr"):
        ip, port, *_ = tr.find_all("td")
        proxies.append(f"{ip.text}:{port.text}")


def free_proxy_list():
    req = get("https://free-proxy-list.net")
    soup = BeautifulSoup(req.text, "html.parser")
    for tr in soup.find("tbody").find_all("tr"):
        ip, port, *_ = tr.find_all("td")
        proxies.append(f"{ip.text}:{port.text}")


def google_proxy():
    req = get("https://www.google-proxy.net/")
    soup = BeautifulSoup(req.text, "html.parser")
    for tr in soup.find("tbody").find_all("tr"):
        ip, port, *_ = tr.find_all("td")
        proxies.append(f"{ip.text}:{port.text}")


def proxyscan():
    req = get("https://www.proxyscan.io/download?type=socks4")
    for proxy in req.text.split():
        proxies.append(proxy)


api_id = 0
api_hash = ""

bot_token = ""

chat_id = -1001411655044
"""
Channel or group id \\
If you want to use id, the id must start with -100 \\
Also you can use username
"""

cron_spec = "*/1 * * * *"
"""
- */1 -> every 1 minute
- */5 -> every 5 minutes \\
and ....
"""

caption = """
**__Count: {}
Time: {}
__**
"""

app = Client(
    'bot',
    api_id,
    api_hash,
    bot_token=""
)


@crontab(cron_spec, start=False)
async def send_proxies():
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    hidemy_name()
    proxyscrape()
    us_proxy()
    free_proxy_list()
    google_proxy()
    proxyscan()

    set_proxies = set(proxies)

    with BytesIO("\n".join(set_proxies).encode()) as file:
        file.name = "proxies.txt"

        await app.send_document(
            chat_id,
            file,
            caption=caption.format(
                len(set_proxies),
                time
            )
        )
        proxies.clear()


app.start()
send_proxies.start()
idle()
send_proxies.stop()
app.stop()
