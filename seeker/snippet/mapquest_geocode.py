#date: 2022-04-05T16:56:45Z
#url: https://api.github.com/gists/40951b456a6e9387ea3478ec0cdc854c
#owner: https://api.github.com/users/creif94

from geopy.geocoders import MapQuest, OpenMapQuest, Nominatim

omapquest_api_key = "APIKEYHERE"

gm = MapQuest(api_key=omapquest_api_key, timeout=60)
gom = OpenMapQuest(api_key=omapquest_api_key, timeout=60)
gn = Nominatim(timeout=60)

loc = "611 West Holt Blvd., ontario, ca, 91762"

l = gom.geocode(loc) or gm.geocode(loc)
print(l.latitude)
print(l.longitude)