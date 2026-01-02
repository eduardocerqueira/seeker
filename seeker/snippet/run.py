#date: 2026-01-02T17:10:41Z
#url: https://api.github.com/gists/6c7470e2b9832765897dc64113577be8
#owner: https://api.github.com/users/ibigio

import requests

url="https://layer.bicyclesharing.net/mobile/v2/fgb/rent"

headers={
  "api-key": "API_KEY_123",
  "authorization": "**********"
}

station_coords = { "lat": 37.7730627, "lon": -122.4390777 }   # from maps
bike_id = "12345"                                             # dummy id

data={
  "userLocation": station_coords,
  "qrCode": { "memberId": "mem123", "qrCode": bike_id},
}

requests.post(url, headers=headers, json=data)ata)