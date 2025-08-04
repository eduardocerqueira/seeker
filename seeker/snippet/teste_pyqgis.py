#date: 2025-08-04T17:06:21Z
#url: https://api.github.com/gists/d24ed01181bc2613109d11d848582c39
#owner: https://api.github.com/users/programandaana

#Visualize Large GeoJSON in QGIS
from qgis.core import QgsVectorLayer
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt # For file modes

# Load the layer (you can use full path or load it from project)
# Optional: Set a starting directory
start_dir = QDir.homePath() 

# Open a file dialog to select a GeoJSON file
file_path, _ = QFileDialog.getOpenFileName(
    None,  # Parent widget (None if not part of a specific widget)
    "Open GeoJSON File",  # Dialog title
    start_dir,  # Starting directory
    "GeoJSON Files (*.geojson *.json);;All Files (*.*)"  # File filter
)

if file_path:
    print(f"Selected GeoJSON file: {file_path}")
    # You can now use 'file_path' to load the GeoJSON data into QGIS
    # e.g., using QgsVectorLayer or other appropriate methods.

layer = QgsVectorLayer(file_path, 'novo_arquivo', 'ogr')

fig, ax = plt.subplots(figsize=(12, 12))

for feature in layer.getFeatures():
    geom = feature.geometry()

    if geom.type() == 0:  # Point or MultiPoint
        if geom.isMultipart():
            points = geom.asMultiPoint()
        else:
            points = [geom.asPoint()]
        for pt in points:
            ax.plot(pt.x(), pt.y(), marker='o', color='black', markersize=2)

    elif geom.type() == 1:  # LineString or MultiLineString
        if geom.isMultipart():
            lines = geom.asMultiPolyline()
        else:
            lines = [geom.asPolyline()]
        for line in lines:
            x = [pt.x() for pt in line]
            y = [pt.y() for pt in line]
            ax.plot(x, y, color='green', linewidth=0.5)

    elif geom.type() == 2:  # Polygon or MultiPolygon
        if geom.isMultipart():
            polygons = geom.asMultiPolygon()
        else:
            polygons = [geom.asPolygon()]
        for polygon in polygons:
            for ring in polygon:
                x = [pt.x() for pt in ring]
                y = [pt.y() for pt in ring]
                ax.plot(x, y, color='blue', linewidth=0.5)

ax.set_title("Large GeoJSON/Shape file")
ax.set_aspect('equal')
plt.show()