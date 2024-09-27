#date: 2024-09-27T16:44:51Z
#url: https://api.github.com/gists/efa7a0170c160cd41ca06aff12ce2bd1
#owner: https://api.github.com/users/valexi7

#!/bin/python3

import requests
from requests.exceptions import RequestException
import hashlib
import json
import urllib3
import urllib.parse
import argparse

parser = argparse.ArgumentParser("ZTE tr069 tool")
parser.add_argument("ip", help="Router ip address", default="192.168.8.1", nargs="?")
parser.add_argument("username", help="Router username", default="admin", nargs="?")
parser.add_argument("password", help= "**********"="1234", nargs="?")
parser.add_argument("--login", help="Login method (multi, single)", default="multi", nargs="?")
parser.add_argument("--settr069", help="Set tr069 data from file", default=None, nargs="?")
parser.add_argument("--ledtest", help="Led test for testing post", action='store_true')
args = parser.parse_args()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

s = requests.Session()

class zteRouter:

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"i "**********"p "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********", "**********"  "**********"l "**********"o "**********"g "**********"i "**********"n "**********") "**********": "**********"
        self.login = login
        self.ip = ip
        self.protocol = "http"  # default to http
        self.username = username
        self.password = "**********"
        self.try_set_protocol()
        self.referer = f"{self.protocol}://{self.ip}/"

    def try_set_protocol(self):
        protocols = ["http", "https"]
        for protocol in protocols:
            url = f"{protocol}://{self.ip}"
            try:
                response = requests.get(url, timeout=5, verify=False)
                if response.ok:
                    self.protocol = protocol
                    # print(f"{self.ip} is accessible via {protocol}")
                    return
            except RequestException:
                pass  # If RequestException occurs, try the next protocol
        # print(f"Could not determine the protocol for {self.ip}")

    def hash(self, str):
        return hashlib.sha256(str.encode()).hexdigest()

    def get_LD(self):
        header = {"Referer": self.referer}
        payload = "isTest=false&cmd=LD"
        r = s.get(self.referer + f"goform/goform_get_cmd_process?{payload}&_=", headers=header, data=payload, verify=False)
        return r.json()["LD"].upper()

    def getVersion(self):
        header = {"Referer": self.referer}
        payload = "isTest=false&cmd=wa_inner_version"
        r = s.get(self.referer + f"goform/goform_get_cmd_process?{payload}", headers=header, data=payload, verify=False)
        return r.json()["wa_inner_version"]


    def get_AD(self):
        print("Calculating AD value")
        def md5(s):
            m = hashlib.md5()
            m.update(s.encode("utf-8"))
            return m.hexdigest()

        def sha256(s):
            m = hashlib.sha256()
            m.update(s.encode("utf-8"))
            return m.hexdigest().upper()  # .upper() to match your example hash

        wa_inner_version = self.getVersion()
        if wa_inner_version == "":
            return ""

        is_mc888 = "MC888" in wa_inner_version
        is_mc889 = "MC889" in wa_inner_version

        hash_function = sha256 if is_mc888 or is_mc889 else md5

        cr_version = ""  # You need to define or get cr_version value as it's not provided in the given code

        a = hash_function(wa_inner_version + cr_version)

        header = {"Referer": self.referer}
        try:
            rd_response = s.get(self.referer + "goform/goform_get_cmd_process?isTest=false&cmd=RD", headers=header, verify=False)
            rd_json = rd_response.json()
            u = rd_json.get("RD", "")

            result = hash_function(a + u)
            print(f"AD: {result}")
            return result
        except Exception as e:
            print(f"Failed to calculate AD: {e}")
            return ""

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"C "**********"o "**********"o "**********"k "**********"i "**********"e "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********", "**********"  "**********"L "**********"D "**********", "**********"  "**********"l "**********"o "**********"g "**********"i "**********"n "**********") "**********": "**********"
        header = {"Referer": self.referer}
        hashPassword = "**********"
        ztePass = "**********"

        if login == "multi":
            payload = {
                'isTest': 'false',
                'goformId': 'LOGIN_MULTI_USER',
                'password': "**********"
                'user': username
            }
        else:
            payload = {
                'isTest': 'false',
                'goformId': 'LOGIN',
                'password': "**********"
            }

        r = s.post(self.referer + "goform/goform_set_cmd_process", headers=header, data=payload, verify=False)
        return "stok=" + r.cookies["stok"].strip('\"')

    def getTRInfo(self):
        ip = self.ip
        cookie = "**********"=self.username, password=self.password, LD=self.get_LD(), login=self.login)

        headers = {
            "Host": ip,
            "Referer": f"{self.referer}index.html",
            "Cookie": f"{cookie}"
        }

        payload = {
            'isTest': 'false',
            'multi_data': '1',
            'cmd': "**********"
        }

        print(payload)

        response = s.get(self.referer + "goform/goform_get_cmd_process", headers=headers, params=payload, verify=False)
        return response.text

    def setTRInfo(self, tr069_data):
        ip = self.ip
        cookie = "**********"=self.username, password=self.password, LD=self.get_LD(), login=self.login)

        headers = {
            "Host": ip,
            "Referer": f"{self.referer}index.html",
            "Cookie": f"{cookie}"
        }
        print(tr069_data["tr069_ServerURL"])
        payload = {
            'isTest': 'false',
            'goformId': 'setTR069Config',
            'AD': self.get_AD(),
            'cr_version': tr069_data["cr_version"],
            'tr069_ServerURL': tr069_data["tr069_ServerURL"],
            #'tr069_ServerURL': urllib.parse.quote(tr069_data["tr069_ServerURL"]),
            'tr069_CPEPortNo': int(tr069_data["tr069_CPEPortNo"]),
            'tr069_ServerUsername': tr069_data["tr069_ServerUsername"],
            'tr069_ServerPassword': "**********"
            'tr069_ConnectionRequestUname': tr069_data["tr069_ConnectionRequestUname"],
            'tr069_ConnectionRequestPassword': "**********"
            #'wan_ipaddr': tr069_data["wan_ipaddr"],
            'tr069_PeriodicInformEnable': int(tr069_data["tr069_PeriodicInformEnable"]),
            'tr069_PeriodicInformInterval': int(tr069_data["tr069_PeriodicInformInterval"]),
            'tr069_CertEnable': tr069_data["tr069_CertEnable"],
            'tr069_DataModule': tr069_data["tr069_DataModule"],
            'tr069_Webui_DataModuleSupport': tr069_data["tr069_Webui_DataModuleSupport"]
        }
        print(payload)

        response = s.post(self.referer + "goform/goform_set_cmd_process", headers=headers, data=payload, verify=False)
        return response.text
    
    def ledTest(self):
        ip = self.ip
        cookie = "**********"=self.username, password=self.password, LD=self.get_LD(), login=self.login)

        headers = {
            "Host": ip,
            "Referer": f"{self.referer}index.html",
            "Cookie": f"{cookie}"
        }

        payload = {
            'isTest': 'false',
            'multi_data': '1',
            'cmd': 'night_mode_switch,night_mode_start_time,night_mode_end_time,night_mode_close_all_led,ODU_led_switch,ODU_led_off_time'
            }

        response = s.get(self.referer + "goform/goform_get_cmd_process", headers=headers, params=payload, verify=False)
        print(response.text)

        payload = {
            'goformId': 'ODU_LED_SWITCH_SET',
            'isTest': 'false',
            'ODU_led_switch': '0',
            'AD': self.get_AD()
        }

        response = s.post(self.referer + "goform/goform_set_cmd_process", headers=headers, data=payload, verify=False)
        return response.text

def load_tr069_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    zteInstance = "**********"

    if args.settr069:
        tr069_data = load_tr069_data(args.settr069)
        print(zteInstance.setTRInfo(tr069_data))

    elif args.ledtest:
        print(zteInstance.ledTest())
    else:
        gatheredJson = json.loads(zteInstance.getTRInfo())
        print(json.dumps(gatheredJson, indent=4))
        with open('tr069.json', 'w') as f:
            json.dump(gatheredJson, f, indent=4)

if __name__ == "__main__":
    main()