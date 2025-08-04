#date: 2025-08-04T17:13:53Z
#url: https://api.github.com/gists/f8409bf4ee0df7d0d84d2456f45c55bb
#owner: https://api.github.com/users/programandaana

#Generate KMZ with images of external source
from osgeo import gdal
import os
from pathlib import Path
import glob
import simplekml
from zipfile import ZipFile

kml = simplekml.Kml()
kml.document.name = "document" #insert name of your document
cwd_os = os.getcwd()
cwd_pathlib = Path.cwd()
python_files = glob.glob("*.tif")
for file in python_files:
    doc_path = os.path.join(cwd_os, file)
    ds = gdal.Open(doc_path)
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx, xres, xskew, miny, yskew, yres  = ds.GetGeoTransform()
    maxx = minx + (ds.RasterXSize * xres)
    maxy = miny + (ds.RasterYSize * yres)
    ground_overlay = kml.newgroundoverlay(name=file)
    ground_overlay.icon.href=file
    ground_overlay.latlonbox.north = maxy
    ground_overlay.latlonbox.south = miny
    ground_overlay.latlonbox.east = maxx
    ground_overlay.latlonbox.west = minx
kml_file = "Test_file.kml" #insert name of your file (temp)
kml.save(kml_file)

    #Create KMZ archive
kmz_file = "document_kmz.kmz" #insert name of your file
with ZipFile(kmz_file, 'w') as myzip:
    myzip.write(kml_file)
    for file in python_files:
        myzip.write(file)
os.remove(kml_file)
print('OK')