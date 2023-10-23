#date: 2023-10-23T16:37:41Z
#url: https://api.github.com/gists/33584271fdbeb2a0c06d616bea23b33e
#owner: https://api.github.com/users/mihalea

import FreeCAD
import Mesh
import os
from PySide import QtGui
from PySide2.QtWidgets import QFileDialog

def import_stl():
    # Check if there is an active document, if not create one
    if FreeCAD.ActiveDocument is None:
        FreeCAD.newDocument()
    else:
        for obj in FreeCAD.ActiveDocument.Objects:
            obj.ViewObject.Visibility = True

    FreeCADGui.setActiveDocument(FreeCAD.ActiveDocument.Name)

    # Prompt user to select STL file
    file_path = QtGui.QFileDialog.getOpenFileName(None, "Select STL file", "", "STL Files (*.stl)")
    
    # Check if user selected a file
    if not file_path[0]:
        return None, None

    # Get the name of the imported STL file without the extension
    stl_filename = os.path.splitext(os.path.basename(file_path[0]))[0]

    # Import STL file
    Mesh.insert(file_path[0], FreeCAD.ActiveDocument.Name)
    
    # Get the last object (the imported STL)
    imported_stl = FreeCAD.ActiveDocument.Objects[-1]
    
    # Select the imported STL
    FreeCADGui.Selection.addSelection(imported_stl)

    return stl_filename, imported_stl

def mesh_to_shape(mesh):
    # Create shape from mesh
    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh.Mesh.Topology, 0.1)  # The second argument is the sewing tolerance

    # Add the shape to the document
    shape_feature = FreeCAD.ActiveDocument.addObject("Part::Feature", "Shape")
    shape_feature.Shape = shape

    # Create a copy and refine shape
    refined_shape = shape_feature.Shape.removeSplitter()
    refined_shape_feature = FreeCAD.ActiveDocument.addObject("Part::Feature", "Refined_Shape")
    refined_shape_feature.Shape = refined_shape

    return refined_shape

def shape_to_solid(shape):
    # Convert refined shape to a solid
    solid = Part.makeSolid(shape)
    solid_feature = FreeCAD.ActiveDocument.addObject("Part::Feature", "Solid")
    solid_feature.Shape = solid

    return solid, solid_feature

def export_to_step(filename, solid_feature):
    # Get the path to the current user's downloads directory
    downloads_dir = os.path.expanduser('~/Downloads')

    # Prompt user for name and location to save the solid as a STEP file
    step_file_path = QtGui.QFileDialog.getSaveFileName(None, "Save Solid as STEP file", os.path.join(downloads_dir,f"{filename}.step"), "STEP Files (*.step)")
        
    # Check if user selected a location
    if step_file_path[0]:
        # Export the solid as a STEP file
        Part.export([solid_feature], step_file_path[0])

def change_view():
    # Hide all objects except the solid
    for obj in FreeCAD.ActiveDocument.Objects:
        obj.ViewObject.Visibility = False
    FreeCAD.ActiveDocument.Objects[-1].ViewObject.Visibility = True

    # Change view to isometric
    FreeCADGui.ActiveDocument.ActiveView.viewIsometric()
    FreeCADGui.ActiveDocument.ActiveView.fitAll()

# Run the function
filename, mesh = import_stl()
if filename:
    shape = mesh_to_shape(mesh)
    solid, solid_feature = shape_to_solid(shape)
    change_view()
    export_to_step(filename, solid_feature)