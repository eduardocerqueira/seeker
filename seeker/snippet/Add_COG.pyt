#date: 2023-01-23T16:44:09Z
#url: https://api.github.com/gists/56b4967689e08db1ccca1d24213eb6e5
#owner: https://api.github.com/users/wiringa

# -*- coding: utf-8 -*-

import os
import arcpy
from osgeo import gdal


class Toolbox(object):
    def __init__(self):
        self.label = "AddCOG Toolbox"
        self.alias = "addcog"

        # List of tool classes associated with this toolbox
        self.tools = [AddCOG]


class AddCOG(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Add COG"
        self.description = "Add COG"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = []
        # First parameter
        param = arcpy.Parameter(
            displayName="COG URL",
            name="cog_url",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        params.append(param)

        param = arcpy.Parameter(
            displayName="Local Raster",
            name="local_raster",
            datatype="DERasterDataset",
            parameterType="Derived",
            direction="Output")
        params.append(param)

        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        gdal.UseExceptions()

        cog_url = parameters[0].valueAsText
        if not cog_url.startswith("/vsicurl/"):
            cog_url = f"/vsicurl/{cog_url}"

        layer_name = os.path.join(arcpy.env.scratchFolder, f"{os.path.basename(cog_url)}.vrt")

        # Create a "virtual" raster, aka VRT - https://gdal.org/drivers/raster/vrt.html
        # Note this doesn't download the COG in it's entirety, it just creates a small XML file (.vrt)
        # with the URL
        ds = gdal.OpenEx(cog_url)
        gdal.Translate(layer_name, ds)  

        arcpy.SetParameterAsText(1, arcpy.Raster(layer_name))


        return
