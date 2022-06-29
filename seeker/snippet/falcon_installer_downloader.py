#date: 2022-06-29T17:27:26Z
#url: https://api.github.com/gists/4161421d7238346e507b1349ff60189f
#owner: https://api.github.com/users/YSaxon

from __future__ import print_function

#this should work with both python2 and python3 with no external dependencies
#most of the time it should be able to automatically determine the os

APIKEY=#apikey with SensorDownload-read permissions only
SECRET=#the secret generated with that key

import sys
import os
together=APIKEY+":"+SECRET
if sys.version_info[0]==2:
    import urllib2 as urllib
    apikeyb64=together.encode('base64')
else:
    import urllib.request as urllib
    import base64
    apikeyb64=base64.b64encode(together.encode()).decode()

import json
url='https://api.crowdstrike.com/oauth2/token'
headers = {'Authorization': 'Basic '+apikeyb64,}
data = {'grant_type': 'client_credentials',}
request=urllib.Request(url,headers=headers,data=json.dumps(data).encode("utf-8"))
response=urllib.urlopen(request)
token=json.loads(response.read())['access_token']
response.close()

url="https://api.crowdstrike.com/sensors/combined/installers/v1?sort=release_date%7Cdesc"
headers={"authorization":"bearer "+token}
request=urllib.Request(url,headers=headers)
response=urllib.urlopen(request)
installers_json=json.loads(response.read())['resources']
response.close()

import platform
def get_distro():
    if platform.system()=="Windows":
      return "Windows",""
    if platform.system()=="Darwin":
        return "macOS",""
    pver=platform.version().lower()
    if "debian" in pver or "ubuntu" in pver or os.path.exists("/etc/debian_version"):
        return "Debian","9/10/11" #same installer as ubuntu
    pplat=platform.platform().lower()
    if "amzn1" in pplat:
        return "Amazon Linux", "1"
    if "amzn2.x86_64" in pplat:
        return "Amazon Linux", "2"
    if "amzn2.aarch64" in pplat:
        return "Amazon Linux", "2 - arm64"    
    with open("/etc/os-release") as osrf:
        osr_lines=osrf.readlines()
    try:
        os_id=[l.split("=")[1][:-1].replace('"','').lower() for l in osr_lines if l.startswith("ID")][0]
        osver_id=[int(float(l.split("=")[1][:-1].replace('"',''))) for l in osr_lines if l.startswith("VERSION_ID")][0]
        if os_id in ["ol","oracle","rhel","centos"] and osver_id in [6,7,8]:
            return "RHEL/CentOS/Oracle",osver_id
        elif os_id =="sles":
            return "SLES",osver_id
    except:pass
    return "notfound",""
    
thisos,thisver=get_distro()

if thisos=="notfound":
    opsystems=dict()
    for i in installers_json:
        if i['os'] and not (i['os'],i['os_version']) in opsystems:
            opsystems[i['os'],i['os_version']]=f"{i['sha256']}  {i['release_date'].split('T')[0]} {i['version']}  {i['file_type']}  {i['os']} {i['os_version']}"
    print("\n"+"\n".join(opsystems.values()))
    print("\nos could not be autodetected")
    sha256todownload=input("please paste in the sha256 of the installer you want to download: ")
    todownload=[i for i in installers_json if i['sha256']==sha256todownload][0]
else:
    todownload=[i for i in installers_json if i['os']==thisos and i['os_version']==thisver][0]

url="https://api.crowdstrike.com/sensors/entities/download-installer/v1?id="+todownload['sha256']
headers={"authorization":"bearer "+token}
request=urllib.Request(url,headers=headers)
response=urllib.urlopen(request)

with open("installer."+todownload["file_type"], "wb") as local_file:
            local_file.write(response.read())
response.close()

print("installer saved to "+"installer."+todownload["file_type"])

url="https://api.crowdstrike.com/sensors/queries/installers/ccid/v1"
headers={"authorization":"bearer "+token}
request=urllib.Request(url,headers=headers)
response=urllib.urlopen(request)
cid=json.loads(response.read())['resources'][0]

if todownload['platform']=="linux":
    if(todownload['os'] in ["Debian","Ubuntu"]):
        print("install with: sudo dpkg -i installer.deb")
    elif(todownload['os'] in ["RHEL/CentOS/Oracle","Amazon Linux"]):
        print("install with: sudo yum install installer.deb")
    elif(todownload['os']=="SLES"):
        print("install with: sudo zypper install installer.rpm")
    else:
        print("first install with the installer file")
    print("then configure with: sudo /opt/CrowdStrike/falconctl -s --cid="+cid)
elif todownload['platform']=="mac":
    print("install with: sudo installer -verboseR -package installer.pkg -target /")
    print("then configure with: sudo /Applications/Falcon.app/Contents/Resources/falconctl license "+cid)
elif todownload['platform']=="windows":
    print("install with: installer.exe /install /quiet /norestart CID="+cid)