#date: 2022-12-09T17:10:25Z
#url: https://api.github.com/gists/294d0d0a89db4b4e7a9b1453361d2acf
#owner: https://api.github.com/users/Rudolfhunter

import requests
from requests import get
import json
import os

#Create Record File
if (os.path.exists("record.txt")) == False:
  f = open("record.txt", "x")
  print("File created")
  f.close()
else:
  print("File exists")

#Get IP
ip = get('https://api.ipify.org').text

#Edit These
name = ""
domain = ""
zone_id = ""
email = ""
auth_key = ""

#First API 2 URLS    
post_url = f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records" 
get_url = f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records?type=A&name={name}.{domain}&content={ip}&proxied=false&page=1&per_page=1&order=type&direction=desc&match=all"

#API Payloads
post_payload = {
    "content": f"{ip}",
    "type": "A",
    "name": f"{name}",
    "priority": 10,
    "proxied": False,
    "ttl": 1
}
get_payload = {
    "type": "A",
    "name": f"{name}",
    "proxied": False,
    "match": "all"  
}
put_payload = {
    "content": f"{ip}",
    "type": "A",
    "name": f"{name}.{domain}",
    "proxied": False,
    "ttl": 1  
}
#API Headers
headers = {
    "Content-Type": "application/json",
    "X-Auth-Email": email,
    "X-Auth-Key": auth_key
}

#Create Record    
def post_function():
  post_response = requests.request("POST", post_url, json=post_payload, headers=headers)
  if post_response.json()['success'] == True:
    print("Post Success: True")
    f = open("record.txt", "w")
    f.write(f"{name} record created.")
    f.close()
  else:
    print("Post Success: False")


#Get Record ID
def get_function():
  get_response = requests.request("GET", get_url, json=get_payload, headers=headers)
  for item in get_response.json()['result']:
    id = item["id"]
  if get_response.json()['success'] == True:
    print("Get Success: True")
  else:
    print("Get Success: False")
  return id
 
#Update Record
def put_function(id):
  put_url = f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records/{id}"
  put_response = requests.request("PUT", put_url, json=put_payload, headers=headers)
  if put_response.json()['success'] == True:
    print("Put Success: True")
    f = open("oldip.txt", "w")
    f.write(f"{ip}")
    f.close()
  else:
    print("Put Success: False")


#Call
f = open("record.txt", "r")
if f.read() == f"{name} record created.":
  f.close()
  print("Record already exists")
  f = open("oldip.txt", "r")
  if f.read() == ip:
    print("IP is up to date")
  else:
    id = get_function()
    put_function(id)
else:
  post_function()
  id = get_function()
  put_function(id)