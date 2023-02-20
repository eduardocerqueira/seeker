#date: 2023-02-20T16:42:57Z
#url: https://api.github.com/gists/823eebc732da7d60487e2227beb73e46
#owner: https://api.github.com/users/joebullardPA

import arcpy
import os

gpkg = r"C:\Users\CLJB3\Documents\Projects\BBS template\v03\bbs_template.gpkg"
walk = arcpy.da.Walk(gpkg)
outGdb = "C:\Working\ArcPro\DefaultProject\Default.gdb"

for path, names, fileNames in walk:
    for fName in filenames:
        outTable = fName.split('.')[1]
        print("Converting: ", outTable)
        try:
            arcpy.conversion.FeatureClassToFeatureClass(fName, outGdb, outTable)
        except arcpy.ExecuteError:
            arcpy.conversion.TableToTable(fName, outGdb, outTable)