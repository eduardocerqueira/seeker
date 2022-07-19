#date: 2022-07-19T17:12:46Z
#url: https://api.github.com/gists/6fe19aaceff227823e6d3bc96956b3c1
#owner: https://api.github.com/users/funcy2267

# Get API Key: https://developers.google.com/maps/documentation/geolocation/get-api-key
# Install required modules: pip3 install access_points
# Example usage: python3 wifi_location.py "YOUR_API_KEY_HERE" --ip

import argparse
import json
import urllib3
from access_points import get_scanner

parser = argparse.ArgumentParser(description='Estimate your location using Wi-Fi networks and Google Maps Geolocation API.')
parser.add_argument('api_key', help='API key for Google Maps Platform')
parser.add_argument('--device', '-d', help='Wi-Fi device to use')
parser.add_argument('--ip', action='store_true', help='Consider your IP in location request')
args = parser.parse_args()

MAPS_URL = "https://www.google.com/maps/@"

def getWifi(device=None):
    if device == None:
        wifi_scanner = get_scanner()
    else:
        wifi_scanner = get_scanner(device)
    wifi_networks = wifi_scanner.get_access_points()
    body = {"wifiAccessPoints": []}
    print("Wi-Fi networks detected:")
    for network in wifi_networks:
        print(network["ssid"])
        body["wifiAccessPoints"] += [{"macAddress": network["bssid"], "signalStrength": network["quality"]*-1}]
    print('')
    return(body)

def getGps(apiKey, networks, consider_ip):
    http = urllib3.PoolManager()
    if consider_ip:
        print("IP Address:", http.request('GET', "https://ipecho.net/plain").data.decode('utf-8')+'\n')
    body = {**{"considerIp": consider_ip}, **networks}
    url = "https://www.googleapis.com/geolocation/v1/geolocate?key="+apiKey
    payload = json.dumps(body)
    headers = {'content-Type': 'application/json', 'Accept-Charset': 'UTF-8'}
    request = http.request('POST', url, headers=headers, body=payload)
    return(request.data.decode('utf-8'))

location = json.loads(getGps(args.api_key, getWifi(args.device), args.ip))
if "error" not in location:
    print("API response:", location)
    print("Google Maps link:", MAPS_URL+str(location["location"]["lat"])+","+str(location["location"]["lng"])+","+str(location["accuracy"])+"z")
else:
    print("Error:", location)
