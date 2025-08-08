#date: 2025-08-08T16:48:24Z
#url: https://api.github.com/gists/88d463f0fa1f8068318d73fff57f74dd
#owner: https://api.github.com/users/datavudeja

def round_to_magnitude(vector):
    def _round_to_digits(a, magnitude, nsig = 2):
	""" round to 'nsig' significant digits """
	return np.around(a / np.power(10, magnitude), nsig - 1) * np.power(10, magnitude)

    order_of_magnitude = np.floor(np.log10(vector))
    func = np.vectorize(_round_to_digits	)
    vector_round = func(vector, order_of_magnitude)
    return vector_round


##############################################################################
# Debug tricks
##############################################################################
# Magical inline to auto-reload python modules
%reload_ext autoreload
%autoreload 2

# Filter warnings as error
import warnings
warnings.filterwarnings("error")


##############################################################################
# Fewer lines running mean
##############################################################################
def running_mean(x,N):
    cumsum = numpy.cumsum(numpy.insert(x,0,0))
    return (cumsum[N:]-cumsum[:-N])/N


##############################################################################
# Time my scripts
##############################################################################
import time
start = time.time()
# "the code you want to test stays here"
end = time.time()
print(end - start)


# A more elegant approach using decorators
# https://realpython.com/primer-on-python-decorators/
import functools
import time

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

@timer
def waste_some_time(num_times):
    for _ in range(num_times):
        sum([i**2 for i in range(10000)])

##############################################################################
# System related
##############################################################################
# Run system command
os.system('command')

# Check memory usage
# https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python
import os
import psutil

pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)

# Check hostname
import socket
socket.gethostname()

# Check username
import getpass
getpass.getuser()

##############################################################################
# xarray operations
##############################################################################
# Create a list of strings of variable names
import xarray as xr
hr = xr.open_dataset('file.nc')
list_of_vars = list(hr.keys())
hr.close()

# Weight data array using latitude: assume dimension [time, lat, lon]
import xarray as xr
import numpy as np

data = xr.open_dataset('file.nc')
wt = np.broadcast_to(np.cos(data.lat.values).reshape(1, -1, 1), data['variable'].shape)
wt = np.ma.masked_array(wt, mask = np.isnan(data['variable']))
data['variable'] = data['variable'] * wt / wt.mean()
... # more analysis
data.close()

# Create a dataset of multiple data arrays
xr.Dataset({'a': (['lat','lon'], a), 'b': (['lat','lon'], b)},
           coords = {'lat': lat, 'lon': lon}, dims = ['lat', 'lon'])

# Decode the "month since " in python; use 'MS' for month start
from dateutil.relativedelta import relativedelta
def decode_month_since(time):
    ref = pd.Timestamp(time.attrs['units'].split(' ')[2])
    start = ref + relativedelta(months = time.values[0])
    start = start.replace(day = 1, hour = 0, minute = 0, second = 0)
    return pd.date_range(start, periods = len(time), freq = 'MS')

# Apply unit function on xarray
myfun = np.argmax # example
da2 = xr.apply_ufunc(myfun, da, input_core_dims = [['time']], 
		     vectorize = True, dask = 'allowed')

# Decode the CF time to pandas DateTimeIndex
pd.to_datetime(xr.decode_cf(hr)['time'].to_index().strftime('%Y-%m-%d'))

# Flip longitude from 0-360 to -180 to 180
def flip_lon(da):
    da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
    da = da.roll(lon = sum(da.lon.values < 0), roll_coords=True)
    return da

# Resample variable to seasonal averages.
def subset_season(var, season):
    if season == 'Annual':
        var_mean = var.groupby('time.year').mean() # sum(min_count = 1)
	var_mean = var_mean.rename({'year': 'time'})
    else:
	var_mean = var.resample(time = 'QS-DEC').mean()
        if season == 'DJF':
            var_mean = var_mean[::4, :, :]
            var_mean[0, :, :] = np.nan
            var_mean = var_mean[:-1, :, :]
        elif season == 'MAM':
            var_mean = var_mean[1::4, :, :]
        elif season == 'JJA':
            var_mean = var_mean[2::4, :, :]
        elif season == 'SON':
            var_mean = var_mean[3::4, :, :]
        var_mean['time'] = var_mean['time'].to_index().year
    return var_mean

# transform between 2D and 3D array
def reshape_3dvar(var):
    temp = var.values.reshape(var.shape[0], -1)
    retain = np.where(~np.isnan(temp[0, :]))[0]
    return retain, temp[:, retain]

def restore_3dvar(var2d, retain, coords):
    var_restored = np.full([len(coords['time']),
                            len(coords['lat']) * \
                            len(coords['lon'])], np.nan)
    var_restored[:, retain] = var2d
    var_restored = xr.DataArray(var_restored,
                                dims = ['time','lat','lon'],
                                coords = coords)
    return var_restored

##############################################################################
# pandas operations
##############################################################################
# Check the basic information of the pandas dataframe/series
df.info() # returns a table of column name, number of non-null (by "null" means "np.nan" as well), and dtypes
df.dtypes
df.describe() # basic statistics on the columns

ds.unique() # returns array of unique values
ds.nunique() # returns the number of unique values
ds.value_counts() # returns the count of each unique value

# To read dates in .csv file, set below in read_csv()
pd.read_csv(..., parse_dates = True)

# Read excel file
pd.read_excel('filename.xlsx', sheetname = 'xxx', index_col = [0,1,2], skiprows=0)

# Create a multiindex from tuples
tuples = [(1, u'red'), (1, u'blue'), (2, u'red'), (2, u'blue')]
pd.MultiIndex.from_tuples(tuples, names = [('number', 'color')])

# Create a multiindex from cross-product of two arrays
numbers = [0, 1, 2]
colors = [u'green', u'purple']
pd.MultiIndex.from_product([numbers, colors], names=['number', 'color'])

# Obtain the labels (has duplicates) at one level from a multiindex 
# (as a single Index object) can use either index or name of the level
mi = pd.MultiIndex.from_arrays((list('abc'), list('def')), names = ['level_1', 'level_2'])
mi.get_level_values(0)
mi.get_level_values('level_1')

# Move one of the multiindex levels to column header
df = pd.DataFrame(data = np.random.randint(10, size=(6,2)),
                  index = pd.MultiIndex.from_product([[0,1,2],['green','blue']], 
                                                     names=['number', 'color'],
                                                     sortorder=2),
                  columns = ['val_1', 'val_2'])
# move 'color' to the columns; the original columns are promoted to top level in
# multi-level column index
df.unstack(level='color')
# move the column headers to the last level in row index
df.stack(level=-1)

# Concatenate columns
df.concat([df1, df2, ...], axis = 1, join = 'outer'/'inner')

# Run moving average
def moving_average(df, w = 48):
    df_temp = df.apply(np.convolve, axis = 0,
                       args = (np.ones(w) / w, 'valid'))
    df_temp.index = df.index[:(-w+1)]
    return df_temp


##############################################################################
# Time alias
##############################################################################
# Most often used: 
Alias   Description
D       calendar day frequency
W       weekly frequency
M       month end frequency
MS      month start frequency
Q       quarter end frequency
QS      quarter start frequency
A       year end frequency
AS      year start frequency
H       hourly frequency
T, min  minutely frequency
S       secondly frequency
# The rest:
B       business day frequency
C       custom business day frequency (experimental)
BM      business month end frequency
CBM     custom business month end frequency
BMS     business month start frequency
CBMS    custom business month start frequency
BQ      business quarter endfrequency
BQS     business quarter start frequency
BA      business year end frequency
BAS     business year start frequency
BH      business hour frequency
L, ms   milliseonds
U, us   microseconds
N       nanoseconds

##############################################################################
# (pandas) Datetime operations
##############################################################################
# month abbreviations
month_name = list('JFMAMJJASOND') # parse to a list
month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
	   
# get the first weekday and number of days in a month
import calendar
wkd_first, days_in_month = calendar.monthrange(dt.year, dt.month)

# group by a pandas data frame by month and, e.g., get the sum
data.groupby([lambda x: x.year, lambda x: x.month]).sum()

# convert string index to datetime-objects index
data.index = pd.to_datetime(data.index, format = '%Y-%m-%d')

# xarray time index to season, beginning in Dec of each year
ds.time.to_index().to_period(freq='Q-NOV')

# number of days in a month
ds.time.to_index().days_in_month

# number of days in a year
time = ds.time.to_index().year
ndays = np.array([int((np.datetime64(str(x+1)+'-01-01') - \
                       np.datetime64(str(x)+'-01-01')) / \
                       np.timedelta64(1, 'D')) for x in ytmp]
# ---- test
time = pd.date_range('1948-01-01', '2000-12-31', freq = '1Y')
ndays = np.array([int((np.datetime64(str(x+1)+'-01-01') - \
                       np.datetime64(str(x)+'-01-01')) / \
                       np.timedelta64(1, 'D')) for x in time.year]

##############################################################################
# Random sampling related
##############################################################################
# Default random sampling
import random
random.seed(999)
# Return a random integer N such that a <= N <= b
random.randint(0,100)

# Use sampling in pandas, e.g. to create training/validation dataset
import pandas as pd
import numpy as np
# ---- note that the default random.seed() is unable to set the random state
#      for this pandas function. Need to input an int/numpy RandomState object
# ---- sample a fraction of the dataset, default without replacement
pd.DataFrame.sample(frac = 0.1, random_state = np.random.RandomState(999))
# ---- sample a fixed number of entries, with replacement
pd.DataFrame.sample(n = 500, replace = True, 
                    random_state = np.random.RandomState(999))


##############################################################################
# Two code snippets testing multiprocessing. Apply_async does not seem to work
# well if large arrays are passed.
##############################################################################
import multiprocessing as mp
import time

# Snippet 1
def dummyfunc(i, j):
    print(i + j)
    return i
pool = mp.Pool(min(mp.cpu_count(), lat_bnd.shape[0]))
for i in range(lat_bnd.shape[0]):
    pool.apply_async(dummyfunc, args = (i, 10),
                     callback = callback)
pool.close()
pool.join()


#Snippet 2
# https://stackoverflow.com/questions/44322268/python-multiprocessing-apply-async-seems-to-be-running-jobs-in-series
def doubler(number):
    time.sleep(1)

def double_trouble(number):
    time.sleep(1)

start_time = time.time()
pool = Pool(processes=10)

for i in range(10):
	pool.apply_async(double_trouble, args = (i,))
	pool.apply_async(doubler, args = (i, ))

pool.close()
pool.join()
print ("We took second: ", time.time()-start_time)

##############################################################################
# Download document from web
# https://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
##############################################################################
import urllib.request
urllib.request.urlretrieve ("http://www.example.com/songs/mp3.mp3", "mp3.mp3")


##############################################################################
# File operations
##############################################################################
# Return a list containing the names of the entries in the directory given
# by path. The list is in arbitrary order, and does not include the special
# entries '.' and '..' even if they are present in the directory.
os.listdir('')

# Check if path exists (either file or directory).
os.path.exists('')

# Check if path is directory.
os.path.isdir('')

##############################################################################
# Numpy array operations
##############################################################################
# Create an empty array
X = np.empty(shape = [0, 2])
# Append values to an array
# https://stackoverflow.com/questions/568962/how-do-i-create-an-empty-array-matrix-in-numpy
for i in range(5):
  for j in range(2):
    X = np.append(X, [[i,j]], axis=0)

# Unique by row
a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
np.unique(a, axis=0)

# Remove singleton dimensions
np.squeeze(a, axis=0)

# Masked array.
x = np.ma.array([3,4,999,7])
x_masked = np.ma.masked_equal(x, value = 999)
	 
##############################################################################
# Geospatial analysis
##############################################################################
# -----------------------------------------------------------------------------------------
# Create a mask of global ocean using cartopy, rasterio, and geopands
# -----------------------------------------------------------------------------------------
import cartopy.feature as crf
import rasterio as rio
from rasterio import features
import geopandas as gpd
import numpy as np
import xarray as xr

ocean_mask = crf.NaturalEarthFeature('physical', 'ocean', '50m')

geom_list = []
for geom in ocean_mask.geometries():
    geom_list.append(geom)
geom = gpd.GeoSeries(geom_list, crs = 'EPSG:4326')

lat = np.arange(89.75, -89.76, -0.5)
lon = np.arange(-179.75, 179.76, 0.5)

transform = rio.transform.from_origin(-180, 90, 0.5, 0.5)
out_arr = features.rasterize(shapes = geom, out = np.zeros([len(lat), len(lon)], dtype = np.uint8),
                                 transform = transform, fill = 0, default_value = 1)

out_arr = xr.DataArray(out_arr > 0, coords = {'lat': lat, 'lon': lon}, dims = ['lat', 'lon'])
		 
out_arr.plot()
		 
		 
# -----------------------------------------------------------------------------------------
# Smooth an image using Gaussian filter and then pad the lost edges
# sigma in Gaussian filter = how much the center be enhanced
# -----------------------------------------------------------------------------------------
# Apply Gaussian smoothing
sigma = 0.5 # Standard deviation of the Gaussian kernel. You can adjust this value as needed.
data = gaussian_filter(data, sigma=sigma)

# Use convolution to add to the edges
# (since the data is positive: add 1 to ensure 0 is not confused with real data)
data = data + 1
data_temp = np.where(np.isnan(data), 0, data)
for i in range(30):
    kernel = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    data1 = convolve2d(data_temp, kernel, mode = 'full')[1:-1, 1:-1]

    kernel = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    data2 = convolve2d(data_temp, kernel, mode = 'full')[1:-1, 1:-1]

    kernel = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    data3 = convolve2d(data_temp, kernel, mode = 'full')[1:-1, 1:-1]

    kernel = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    data4 = convolve2d(data_temp, kernel, mode = 'full')[1:-1, 1:-1]

    data_temp = np.maximum(np.maximum(np.maximum(data1, data2), data3), data4)

    data_temp = np.where(np.isnan(data), data_temp, data)
data_temp[data_temp < (1-1e-9)] = np.nan

data = np.where(np.isnan(data), data_temp, data)
data = data - 1. # subtract back
