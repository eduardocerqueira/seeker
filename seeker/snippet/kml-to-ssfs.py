#date: 2023-11-03T17:07:05Z
#url: https://api.github.com/gists/a3e7a0b7cb8c65e3a6151c5aa4369548
#owner: https://api.github.com/users/rayslava

import xml.etree.ElementTree as ET
import uuid

# Parse the KML file
tree = ET.parse('yourfile.kml')
root = tree.getroot()

# KML files use the default namespace, define it to search for elements
namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

# Create the root element for the Subsurface-compatible XML
divesites = ET.Element('divesites', attrib={'program': 'subsurface', 'version': '3'})

# Iterate through all Placemark elements in the KML file
for placemark in root.findall('.//kml:Placemark', namespace):
    name = placemark.find('.//kml:name', namespace).text
    point = placemark.find('.//kml:Point/kml:coordinates', namespace)
    
    # Check if the point tag and coordinates are present
    if point is not None:
        # Generate a unique identifier for each site
        site_uuid = str(uuid.uuid4())
        # Assume the coordinates are in the format "longitude,latitude,altitude"
        # and strip any whitespace or newline characters
        coordinates = point.text.strip().split(',')
        gps = f"{coordinates[1]} {coordinates[0]}"  # Reorder to "latitude longitude"

        # Create the site element
        site = ET.SubElement(divesites, 'site', uuid=site_uuid, name=name, gps=gps)
        
        # Add the <geo> element for Philippines to each site
        geo = ET.SubElement(site, 'geo', cat='2', origin='0', value='Philippines')

# Convert the ET.Element into a string of XML
xmlstr = ET.tostring(divesites, encoding='unicode', method='xml')

print(xmlstr)
