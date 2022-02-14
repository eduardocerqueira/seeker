#date: 2022-02-14T17:07:05Z
#url: https://api.github.com/gists/54c835f6df446474241d13cd088069ed
#owner: https://api.github.com/users/drhammed

import numpy as np 

lat_min = 45.0
lat_max = 60.0
lon_min = 85.0
lon_max = 100.0
density = 3 # three points per `squared` degree
import numpy as np 

np.random.seed(30) # for reproducibility, this line could be deleted

number_of_points = (lat_max - lat_min) * (lon_max - lon_min) * density # you can use poisson distribution here to include additional portion of randomness
pseudoabsence_latitudes = np.random.uniform(lat_min, lat_max, number_of_points)
pseudoabsence_longitudes = np.random.uniform(lat_min, lat_max, number_of_points) 

###### This Could be easily converted into function

def get_pseudoabsence_data(lat_min, lat_max, lon_min, lon_max, density=3):
    """Generates pseudoabsence data.
    
    Parameters
    ==========
    
        :param lat_min: 
        :type lat_min: double
        #... fill docs
        :rtype: tuple
        :returns: (lats, lons), where lats and lons are arrays of ps-absence point coordinates
        
    """
    # you can use poisson distribution here to include additional portion of randomness
    number_of_points = (lat_max - lat_min) * (lon_max - lon_min) * density
    pseudoabsence_latitudes = np.random.uniform(lat_min, lat_max, number_of_points)
    pseudoabsence_longitudes = np.random.uniform(lat_min, lat_max, number_of_points) 
    return (pseudoabsence_latitudes, pseudoabsence_longitudes)


# USAGE: 
lats, lons = get_pseudoabsence_data(12, 15, 100, 120, density=3)



