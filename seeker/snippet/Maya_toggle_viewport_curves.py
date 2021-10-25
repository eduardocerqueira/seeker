#date: 2021-10-25T17:16:40Z
#url: https://api.github.com/gists/78ced5927015c0e3f16984105e805a68
#owner: https://api.github.com/users/L0Lock

import maya.cmds as cmds
import maya.OpenMaya as om

def toggle_viewport_curves():
    # gets current panel
    currentPanel = cmds.getPanel(wf=True)
    # checks if its a viewport
    if cmds.getPanel(to=(currentPanel)) != "modelPanel":
        om.MGlobal.displayError("No viewport is active.")
        return False
    
    cmds.modelEditor(currentPanel, e=True, nc=not cmds.modelEditor(currentPanel, q=True, nc=True))
    
    return True
    
toggle_viewport_curves()

# Written by Lo?c "L0Lock" DAUTRY
# Source:
# https://gist.github.com/L0Lock/78ced5927015c0e3f16984105e805a68