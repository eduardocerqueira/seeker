#date: 2023-11-29T16:57:06Z
#url: https://api.github.com/gists/996c074c7730aad8cb4666683cfd78cd
#owner: https://api.github.com/users/jmwright

# import freecad
import os
import zipfile
import tempfile
import cadquery as cq
from cadquery.vis import show

def import_part_static(fc_part_path):
    """
    Imports without parameter handling by extracting the brep file from the FCStd file.
    Does NOT require FreeCAD to be installed.
    Parameters:
        fc_part_path - Path to the FCStd file to be imported.
    Returns:
        A CadQuery Workplane object or None if the import was unsuccessful.
    """

    res = None

    # Make sure that the caller gave a valid file path
    if not os.path.isfile(fc_part_path):
        print("Please specify a valid path.")
        return None

    # A temporary directory is required to extract the zipped files to
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the contents of the file
        with zipfile.ZipFile(fc_part_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Open the file with CadQuery
        res = cq.Workplane(cq.Shape.importBrep(os.path.join(temp_dir, "PartShape.brp")))

    return res


def get_parameters(fc_part_path):
    """
    Extracts the parameters from the spreadsheet inside the FCStd file.
    Does NOT require FreeCAD to be installed.
    Parameters:
        fc_part_path - Path to the FCStd file to be imported.
    Returns:
        A dictionary of the parameters and their initial values.
    """

    # Make sure that the caller gave a valid file path
    if not os.path.isfile(fc_part_path):
        print("Please specify a valid path.")
        return None

    # This will keep the collection of the parameters and their current values
    parameters = {}

    # To split units from values
    import re

    # So that the XML file can be parsed
    import xml.etree.ElementTree as ET

    # A temporary directory is required to extract the zipped files to
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the contents of the file
        with zipfile.ZipFile(fc_part_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # parse the Document.xml file that holds metadata like the spreadsheet
        tree = ET.parse(os.path.join(temp_dir, 'Document.xml'))
        root = tree.getroot()
        objects = root.find('ObjectData')
        for object in objects.iter("Object"):
            if object.get('name') == "Spreadsheet":
                props = object.find('Properties')
                for prop in props.iter("Property"):
                    if prop.get('name') == "cells":
                        for cell in prop.find("Cells").iter():
                            if cell is None or cell.get('content') is None:
                                continue

                            # Determine whether we have a parameter name or a parameter value
                            if "=" not in cell.get('content'):
                                # Make sure we did not get a description
                                if cell.get('address')[0] != "A" and cell.get('address')[0] != "B":
                                    continue

                                # Start a parameter entry in the dictionary
                                parameters[cell.get('content')] = {}
                            elif "=" in cell.get('content'):
                                # Extract the units
                                units = "".join(re.findall("[a-zA-Z]+", cell.get('content')))
                                if units is not None:
                                    parameters[cell.get('alias')]["units"] = units
                                else:
                                    parameters[cell.get('alias')]["units"] = "N/A"

                                # Extract the parameter value and store it
                                value = cell.get('content').replace("=", "").replace(units, "")
                                parameters[cell.get('alias')]["value"] = value
                break
            else:
                continue

        return parameters
        


def main():
    # Used with FreeCAD models here: https://github.com/hoijui/nimble/tree/master/src/mech/freecad

    # Universal shelf
    # res = import_freecad_part_static("models/base_shelf.FCStd")
    # cq.exporters.export(res, "exports/base_shelf.stl")

    # Rail/rack leg
    # res = import_freecad_part_static("models/master_rail.FCStd")
    # cq.exporters.export(res, "exports/master_rail.stl")

    # Raspberry Pi shelf
    # res = import_freecad_part_static("models/rpi_4b_shelf.FCStd")
    # cq.exporters.export(res, "exports/rpi_4b_shelf.stl")

    # if res is not None:
    #     show(res)

    parameters = get_parameters("models/base_shelf.FCStd")
    print(parameters)

if __name__ == "__main__":
    main()
