#date: 2022-08-01T16:56:34Z
#url: https://api.github.com/gists/e43b6eaebcfcb244b25bec571c3debe0
#owner: https://api.github.com/users/d-saikrishna

def create_nc_from_images(lt_s, lt_n, ln_w, ln_e, image_path, image_name):
    '''
    Creates NetCDF4 files from the grayscale image using the BBOX coordinates.

    Input parameters:
    BBOX coordindates (lt_s, lt_n, ln_w, ln_e)
    image_path: filepath of the image to be converted.
    image_name: output file name

    Returns the filepath of the NetCDF4 file created.
    '''
    image = Image.open(image_path)
    grayscale_array = np.asarray(image)

    lt_array = np.linspace(lt_n, lt_s, grayscale_array.shape[0])
    ln_array = np.linspace(ln_w, ln_e, grayscale_array.shape[1])

    my_file = Dataset(r"NCs/" + image_name + '.nc', 'w', format='NETCDF4')
    lat_dim = my_file.createDimension('lat', grayscale_array.shape[0])
    lon_dim = my_file.createDimension('lon', grayscale_array.shape[1])
    time_dim = my_file.createDimension('time', None)

    latitudes = my_file.createVariable("lat", 'f4', ('lat',))
    longitudes = my_file.createVariable("lon", 'f4', ('lon',))
    time = my_file.createVariable('time', np.float32, ('time',))

    new_nc_variable = my_file.createVariable("Inundation", np.float32, ('time', 'lat', 'lon'))
    latitudes[:] = lt_array
    longitudes[:] = ln_array
    new_nc_variable[0, :, :] = grayscale_array

    my_file.close()

    return r"NCs/" + image_name + '.nc'


def create_tiffs_from_ncs(nc_path, image_name):
    '''
    Creates GeoTIFF files from the NetCDF4 files.

    Input parameters:
    nc_path: filepath of the NetCDF4 file.
    image_name: Output name of the GeoTIFF
    '''

    tiff_file = xr.open_dataset(nc_path)
    var = tiff_file['Inundation']
    var = var.rio.set_spatial_dims('lon', 'lat')
    var.rio.set_crs("epsg:4326")
    var.rio.to_raster(r"tiffs/" + image_name + r".tif")
    tiff_file.close()


import glob
extension = 'image'
        #os.chdir(path)
result = glob.glob('Tiles/*.{}'.format(extension))

# Make a list of all latitudes and longitudes from the BBOXs.
lats = []
lons = []
for file in result:
    lats.append(float(file.split(',')[-1].split('.image')[0]))
    lons.append(float(file.split(',')[-2]))

nc_path = create_nc_from_images(starting_point_lat,max(lats),starting_point_lon,max(lons),date_string+'.png',date_string)
tiff = create_tiffs_from_ncs(nc_path,date_string)