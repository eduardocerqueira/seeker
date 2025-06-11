#date: 2025-06-11T17:13:11Z
#url: https://api.github.com/gists/fe134ce1df6963152bfaf1ce6c5a2d24
#owner: https://api.github.com/users/cec12583

import arcpy as ap
import os
import arcpy.mp

ap.env.workspace = input("Add your working directory and press enter: ")
print("Workspace set to:", ap.env.workspace)
ap.env.overwriteOutput = True
from arcpy.sa import*


#Creating the input and output files
input_shape = input("Enter the name of your shapefile [.shp], and press enter: ")
default_tif = "Annual_NLCD_LndCov_2023_CU_C1V0_2.tif"
print("The default raster is the 2023 NCLD land cover classification at 30-meter resolution (Annual_NLCD_LndCov_2023_CU_C1V0_2.tif)")
input_NLCD = input(f"If you're using the NLCD from a different year, enter the name here [.tif] (press Enter to use default: {default_tif}): ")
if input_NLCD.strip() == "":
    input_NLCD = default_tif

clipped_raster = "clipped_raster.tif"
input_raster = ap.Raster(input_NLCD)

symbology = os.path.join(ap.env.workspace, "LandUse_Symbology.lyrx")


#Checking projection
shape_desc = ap.Describe(input_shape)
shape_sr = shape_desc.spatialReference

#Updating missing projections to WSG 1984
if not shape_sr.name or shape_sr.name.lower() == "unknown":
    print("No spatial reference found in shapefile. Defining as WGS 1984.")
    sr = ap.SpatialReference(4326)
    ap.management.DefineProjection(input_shape, sr)
    shape_sr = ap.Describe(input_shape).spatialReference

# #Making sure projections match
raster_sr = ap.Describe(input_NLCD).spatialReference
if shape_sr.name != raster_sr.name:
    print("Reprojecting shapefile to match raster.")
    reprojected_shape = "input_shape_reprojected.shp"
    ap.management.Project(input_shape, reprojected_shape, raster_sr)
    input_shape = reprojected_shape
    shape_sr = ap.Describe(input_shape).spatialReference
    
print("Final shapefile projection:", shape_sr.name)
print("Final raster projection:", raster_sr.name)

#Clipping land cover data to the user's state shapefile
ap.management.Clip(input_raster, "", clipped_raster, input_shape, "NoData", "ClippingGeometry", "MAINTAIN_EXTENT")
clipped_raster_obj = ap.Raster(clipped_raster)

#Removing null values
raster_nodata = SetNull(clipped_raster_obj == 0, clipped_raster_obj)
raster_nodata.save("clean_raster.tif")
clean_raster = "clean_raster.tif"

#Converting to polygon data
landuse_polygon = "land_use.shp"
full_landuse_path = os.path.join(ap.env.workspace, landuse_polygon)
ap.RasterToPolygon_conversion(clean_raster, landuse_polygon, "SIMPLIFY", "VALUE")

fields = ap.ListFields(landuse_polygon)

# Print the name of each field
print("Fields in land use polygon after conversion:")
for field in fields:
    print("-", field.name)

# Adding a field for the land use titles 
ap.AddField_management(landuse_polygon, "LandUse", "TEXT", field_length=50)

#Checking new fields
fields = ap.ListFields(landuse_polygon)
print("Updated fields after adding 'LandUse':")
for field in fields:
    print("-", field.name)

#Adding land use titles 
with ap.da.UpdateCursor(landuse_polygon, ["GRIDCODE", "LandUse"]) as cursor:
    for row in cursor:
        if row[0] == 11:
            row[1] = "Open Water"
        elif row[0] == 12:
            row[1] = "Perennial Ice/Snow"
        elif row[0] == 21:
            row[1] = "Developed, Open Space"
        elif row[0] == 22:
            row[1] = "Developed, Low Intensity"
        elif row[0] == 23:
            row[1] = "Developed, Medium Intensity"
        elif row[0] == 24:
            row[1] = "Developed, High Intensity"
        elif row[0] == 31:
            row[1] = "Barren Land (Rock/Sand/Clay)"
        elif row[0] == 41:
            row[1] = "Deciduous Forest"
        elif row[0] == 42:
            row[1] = "Evergreen Forest"
        elif row[0] == 43:
            row[1] = "Mixed Forest"
        elif row[0] == 51:
            row[1] = "Dwarf Scrub"
        elif row[0] == 52:
            row[1] = "Shrub/Scrub"
        elif row[0] == 71:
            row[1] = "Grassland/Herbaceous"
        elif row[0] == 72:
            row[1] = "Sedge/Herbaceous"
        elif row[0] == 73:
            row[1] = "Lichens"
        elif row[0] == 74:
            row[1] = "Moss"
        elif row[0] == 81:
            row[1] = "Pasture/Hay"
        elif row[0] == 82:
            row[1] = "Cultivated Crops"
        elif row[0] == 90:
            row[1] = "Woody Wetlands"
        elif row[0] == 95:
            row[1] = "Emergent Herbaceous Wetlands"
        else:
            row[1] = "Unknown"
        cursor.updateRow(row)

print("Land use name fields have been added.") 

#Creating a dictionary of RGB codes corresponding to each land use class
land_use_colors = {
    "Barren Land (Rock/Sand/Clay)": [182, 170, 147],
    "Cultivated Crops": [168, 112, 0],
    "Deciduous Forest": [15, 153, 76],
    "Developed, High Intensity": [168, 0, 0],
    "Developed, Low Intensity": [222, 150, 106],
    "Developed, Medium Intensity": [255, 0, 0],
    "Developed, Open Space": [216, 172, 168],
    "Dwarf Scrub": [177, 152, 102],
    "Emergent Herbaceous Wetlands": [102, 153, 205],
    "Evergreen Forest": [0, 100, 29],
    "Grassland/Herbaceous": [242, 237, 193],
    "Lichens": [156, 205, 102],
    "Mixed Forest": [171, 205, 132],
    "Moss": [138, 198, 171],
    "Open Water": [68, 79, 137],
    "Pasture/Hay": [230, 230, 0],
    "Perennial Ice/Snow": [220, 231, 255],
    "Sedge/Herbaceous": [191, 193, 91],
    "Shrub/Scrub": [212, 182, 134],
    "Woody Wetlands": [190, 221, 255],
}


#Adding the RGB codes to a new field
ap.AddField_management(landuse_polygon, "RGB_Code", "TEXT")
fields = ap.ListFields(landuse_polygon)
print("Updated fields after adding 'RGB_Code':")
for field in fields:
    print("-", field.name)

count = ap.management.GetCount(landuse_polygon)
print("Feature count in land_use.shp:", count)

try:
    with ap.da.UpdateCursor(landuse_polygon, ["LandUse", "RGB_Code"]) as cursor:
        for row in cursor:
            land_use = row[0]
            if land_use in land_use_colors:
                color = land_use_colors[land_use]
                row[1] = f"{color[0]},{color[1]},{color[2]}"
            cursor.updateRow(row)
    print("RGB values have been assigned to the 'RGB_Code' field.")
except Exception as e:
    print("error durig RGB update:", e)


# Asking user if they're running the script inside an ArcGIS project
inside_arc = input("Are you running this script inside ArcGIS Pro with the map open? (yes/no): ").strip().lower()

if inside_arc == "yes":
    print("Attempting to apply symbology in ArcGIS Pro ...")
    try:
        ap.management.ApplySymbologyFromLayer(full_landuse_path, symbology)
        print("Symbology applied :)")
    except Exception as e:
        print("Failed to apply symbology:", e)
else:
    print("Skipping symbology. Please apply LandUse_Symbology.lyrx to the landuse.shp layer manually.")


# Option to export PNG (The extent will only work for the San Fransisco area)
export_map = input("Would you like to generate a PNG of your map with symbology applied? (yes/no): ").strip().lower()

if export_map == "yes":
    print("Preparing export...")

    aprx_path = os.path.join(ap.env.workspace, "MapTemplate.aprx")
    output_png = os.path.join(ap.env.workspace, "land_use_map.png")

    try:
        # Step 2: Load the project and map
        aprx = ap.mp.ArcGISProject(aprx_path)
        map_obj = aprx.listMaps()[0]

        # Step 3: Add the symbologized shapefile to the map
        landuse_layer = map_obj.addDataFromPath(full_landuse_path)
        ap.management.ApplySymbologyFromLayer(landuse_layer, symbology)

        # Step 4: Export
        layout = aprx.listLayouts()[0] if aprx.listLayouts() else None
        if layout:
            layout.exportToPNG(output_png, resolution=300)
            print(f"Map exported to: {output_png}")
        else:
            #map_obj.defaultCamera.setExtent(map_obj.getLayerExtent(landuse_layer))
            #map_obj.exportToPNG(output_png, resolution=300)
            print(f"No Layout found.")
    except Exception as e:
        print("Failed to export map:", e)

print("All done!")
