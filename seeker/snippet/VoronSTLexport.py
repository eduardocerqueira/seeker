#date: 2023-03-14T16:52:30Z
#url: https://api.github.com/gists/e8ea733d457af04a606faa9fa5bda02b
#owner: https://api.github.com/users/pecirep

"""
Exports all Bodies of material "ABS Plastic" from a Fusion360 Project to STLs
Assembly structure is maintained as folder structure
files are named according to the Voron standard:
    - prepend '[a]_' if the part is an accent
    - append _x[n] for the required count

if there is only one body in a component, the filename is [componentName].stl
if multiple bodies share a component, the filenames will be [componentName]_[bodyName].stl
"""

import adsk.core, adsk.fusion, traceback, re, os

__author__ = 'Tin Pecirep'
__copyright__ = 'Copyright 2020, VoronDesign'
__credits__ = ['Tin Pecirep']
__license__ = 'GPLv3'
__version__ = '1.0.1'
__maintainer__ = 'Tin Pecirep'
__email__ = 'tin.pecirep@gmail.com'
__status__ = 'Development'

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface

        # Set styles of file dialog.
        fileDlg = ui.createFileDialog()
        fileDlg.isMultiSelectEnabled = False
        fileDlg.title = 'Fusion File Dialog'
        fileDlg.filter = '*.*'
            
        # Set styles of file dialog.
        folderDlg = ui.createFolderDialog()
        folderDlg.title = 'Fusion Folder Dialog' 
        
        # Show folder dialog
        dlgResult = folderDlg.showDialog()
        if dlgResult == adsk.core.DialogResults.DialogOK:
            format(folderDlg.folder)
        
        exportFolder = folderDlg.folder

        product = app.activeProduct
        design = adsk.fusion.Design.cast(product)

        # Get all occurrences in the root component of the active design
        root = design.rootComponent
        occs = root.allOccurrences
        
        # Gather information about each unique component
        bodiesToExport = []
        for occ in occs:
            comp = occ.component
            
            if hasattr(comp.material, 'name') and comp.material.name == 'ABS Plastic': pass
            elif occ.bRepBodies and 'ABS Plastic' in [bdy.material.name for bdy in occ.bRepBodies]: pass
            else: continue
            if len(comp.bRepBodies) == 0: continue # I don't understand this

            jj = 0
            for componentsI in bodiesToExport:
                if componentsI['component'] == comp:
                    # Increment the instance count of the existing row.
                    componentsI['instances'] += 1
                    break
                jj += 1

            if jj == len(bodiesToExport):
                # Generate file location
                treeLocation = re.sub(':\d+\+','/',occ.fullPathName)
                #path = 'C:/temp/VoronSTLExport/' + app.activeDocument.name + '/' + re.sub(':\d+','', treeLocation)
                componentpath = app.activeDocument.name + '/' + re.sub(':\d+','', treeLocation).replace(' ', '_')

                #last item in path is the component itself
                os.makedirs(exportFolder + '/' + '/'.join(componentpath.split('/')[:-1]), exist_ok = True)
                
                # Add all bodies to the bodiesToExport
                bodycount = len(comp.bRepBodies)
                for body in comp.bRepBodies:

                    bodypath = componentpath
                    # prepend underscore if appearance is red
                    if hasattr(comp.material, 'appearance') and 'red' in comp.material.appearance.name.lower(): '/'.join([bodypath.split('/')[:-1], '[a]_' + bodypath.split('/')[-1]])
                    # append name of body if there is more than one body.
                    if bodycount >1: bodypath += '_' + body.name

                    bodiesToExport.append({
                        'body': body,
                        'component': comp,
                        'path': bodypath,
                        'instances': 1
                    })

        # export STLs
        for exportBody in bodiesToExport:
            exportMgr = adsk.fusion.ExportManager.cast(design.exportManager)
            stlOptions = exportMgr.createSTLExportOptions(exportBody['body'])
            #stlOptions.isOneFilePerBody = True
            stlOptions.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
            stlOptions.filename = exportFolder + '/' + exportBody['path'] + '_x' + str(exportBody['instances']) + '.stl'
            exportMgr.execute(stlOptions)

        ui.messageBox(app.activeDocument.name +'Success! Exported ' + str(len(bodiesToExport)) + ' STLs.')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
