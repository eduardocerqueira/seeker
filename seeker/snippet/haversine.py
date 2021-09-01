#date: 2021-09-01T01:34:36Z
#url: https://api.github.com/gists/91145a57115e5ba0e069318465a626dd
#owner: https://api.github.com/users/kenilc

import numpy as np

def haversine_np(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return 6371 * c * 1000  # m.