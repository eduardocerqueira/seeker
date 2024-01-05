#date: 2024-01-05T17:08:54Z
#url: https://api.github.com/gists/db84c6b0bc5f704b648bd4f9a8e75f0a
#owner: https://api.github.com/users/bennyistanto

import requests
from requests.auth import HTTPBasicAuth
import os

# GeoServer details
geoserver_url = "http://localhost:8080/geoserver"
workspace = "your_workspace"
username = "admin"
password = "**********"
data_directory = "path_to_your_data_folder"  # Folder containing .nc and .sld files

def upload_nc_file(nc_file, store_name):
    rest_url = f"{geoserver_url}/rest/workspaces/{workspace}/coveragestores/{store_name}/file.nc?configure=first&coverageName={store_name}"
    headers = {"Content-type": "application/x-netcdf"}
    
    with open(nc_file, 'rb') as data:
        response = "**********"=headers, data=data, auth=HTTPBasicAuth(username, password))
    
    if response.status_code == 201:
        print(f"NetCDF file {nc_file} uploaded successfully")
    else:
        print(f"Error uploading NetCDF file {nc_file}: {response.content}")

def upload_sld(sld_file, style_name):
    sld_url = f"{geoserver_url}/rest/styles"
    sld_headers = {"Content-type": "application/vnd.ogc.sld+xml"}
    sld_data = f"<style><name>{style_name}</name><filename>{sld_file}</filename></style>"

    response = "**********"=sld_headers, data=sld_data, auth=HTTPBasicAuth(username, password))

    if response.status_code == 201:
        with open(sld_file, 'r') as file:
            sld_content = file.read()
            style_url = f"{geoserver_url}/rest/styles/{style_name}"
            response = "**********"=sld_headers, data=sld_content, auth=HTTPBasicAuth(username, password))
            if response.status_code == 200:
                print(f"SLD file {sld_file} uploaded successfully")
            else:
                print(f"Error uploading SLD file {sld_file}: {response.content}")
    else:
        print(f"Error creating SLD style for {sld_file}: {response.content}")

def associate_style(layer_name, style_name):
    layer_url = f"{geoserver_url}/rest/layers/{workspace}:{layer_name}"
    xml = f"<layer><defaultStyle><name>{style_name}</name></defaultStyle></layer>"
    headers = {"Content-type": "text/xml"}

    response = "**********"=headers, data=xml, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        print(f"Style {style_name} associated with NetCDF layer {layer_name} successfully")
    else:
        print(f"Error associating style {style_name}: {response.content}")

# Scan the directory for NetCDF files and process each one
for file in os.listdir(data_directory):
    if file.endswith(".nc"):
        nc_file = os.path.join(data_directory, file)
        sld_file = os.path.join(data_directory, file.replace(".nc", ".sld"))
        store_name = file[:-3]  # Remove .nc extension for store name
        
        if os.path.exists(sld_file):
            upload_nc_file(nc_file, store_name)
            upload_sld(sld_file, store_name)
            associate_style(store_name, store_name)
        else:
            print(f"No matching SLD file found for {nc_file}")
    else:
            print(f"No matching SLD file found for {nc_file}")
