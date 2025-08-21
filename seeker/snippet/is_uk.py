#date: 2025-08-21T17:19:10Z
#url: https://api.github.com/gists/f6e89ab95402f35b3481987626373033
#owner: https://api.github.com/users/jasoncartwright

def is_uk(lat_lng):

    lat = float(lat_lng.split(",")[0])
    lng = float(lat_lng.split(",")[1])

    sw_lat = 49.1
    sw_lng = -14.015517
    ne_lat = 61.061
    ne_lng = 2.0919117

    if lat < sw_lat: return False
    if lng < sw_lng: return False
    if lat > ne_lat: return False
    if lng > ne_lng: return False

    return True