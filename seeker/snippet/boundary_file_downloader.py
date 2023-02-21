#date: 2023-02-21T16:52:04Z
#url: https://api.github.com/gists/d15fb6fbca81c27d7382d51416d0c83c
#owner: https://api.github.com/users/maxdgt

import os
import io
import requests
import zipfile

def boundary_file_downloader(year=2019, state='us', entity='state', resolution='500k', filetype='shp', path=None):
    """Downloads US Census Bureau Cartographoc Boundary files to a dedicated folder.

    Args:
        year (int, optional): Year the data should be pulled from. Defaults to 2019.
        state (str, optional): The state FIPS code, or 'us' for national level. Defaults to 'us'.
        entity (str, optional): The entity to be download. A list can be found 
            at https://www2.census.gov/geo/tiger/GENZ2019/2019_file_name_def.pdf. Defaults to 'state'.
        resolution (str, optional): Resolution of the data; 500k, 5m, or 20m. Defaults to '500k'.
        filetype (str, optional): SHP or KML. Defaults to 'shp'.
        path (str, optional): Directory to save files to. Defaults to None.
    """    
    folder = '_'.join(['cb',
                         str(year),
                         state,
                         entity,
                         resolution])
    filename = folder + '.zip'
    base = 'https://www2.census.gov/geo/tiger'
    folder_year = 'GENZ' + str(year)
    file_type = filetype
    query = [base, folder_year, file_type, filename]
    url = '/'.join(query)
    
    
    if path:
        folder = os.path.join(path, folder)

    os.mkdir(folder)

    r = requests.get(url)
    if r.ok:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(folder)
        print(f'Downloaded and extracted {url}')   

    else:
        print(f'Request for {url} failed')   
