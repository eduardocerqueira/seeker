#date: 2022-05-11T17:18:49Z
#url: https://api.github.com/gists/4c47e0d51c54c8fa72f7a22648f0c2e7
#owner: https://api.github.com/users/Adarshreddyash

"""
Find user/database entries within a km radius based on long/lat co-ords.
i.e. return all objects where longlat lies within 10km of my current long/lat.
Using with Django REST Framework but approach is same for any similar req.
"""
import math

def get_queryset(self):
    user = self.request.user
    lat = self.request.query_params.get('lat', None)
    lon = self.request.query_params.get('long', None)

    if lat and lon:
        lat = float(lat)
        lon = float(lon)

        # Haversine formula = https://en.wikipedia.org/wiki/Haversine_formula
        R = 6378.1  # earth radius
        bearing = 1.57  # 90 degrees bearing converted to radians.
        distance = 10  # distance in km

        lat1 = math.radians(lat)  # lat in radians
        long1 = math.radians(lon)  # long in radians

        lat2 = math.asin(math.sin(lat1)*math.cos(distance/R) +
                         math.cos(lat1)*math.sin(distance/R)*math.cos(bearing))

        long2 = long1 + math.atan2(math.sin(bearing)*math.sin(distance/R)*math.cos(lat1),
                                   math.cos(distance/R)-math.sin(lat1)*math.sin(lat2))

        lat2 = math.degrees(lat2)
        long2 = math.degrees(long2)

        queryset = Location.objects.filter(current_lat__gte=lat1, current_lat__lte=lat2)\
            .filter(current_long__gte=long1, current_long__lte=long2)
