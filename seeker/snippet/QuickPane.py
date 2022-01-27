#date: 2022-01-27T17:06:34Z
#url: https://api.github.com/gists/8490b8c96511c1086db1c5e8da8b1e10
#owner: https://api.github.com/users/ggnkua

#------------------------------------------------------------------------
import os
import N10X

#------------------------------------------------------------------------
# Stick this file in your 10x PythonScripts folder,
# typically \users\USERNAME\AppData\Roaming\10x\PythonScripts
# For this to work, disable F1 and F2 keys in KeyMappings.10x_settings first!

#------------------------------------------------------------------------
# enable/disable this script here
QuickPaneEnabled = True

#------------------------------------------------------------------------
# Called when a key is pressed.
# Return true to surpress the key
def OnInterceptKey(key, shift, control, alt):
	
    # Kill all panels as we need the horizontal space
    if key == "F1":
        SetColumnCount1()
        return True

    # Increase column count and duplicate current panel into the other
    if key == "F2":
        SetColumnCount2()
        DuplicatePanelRight()
        return True
        
#------------------------------------------------------------------------
def Initialise():
	N10X.Editor.RemoveUpdateFunction(Initialise)
	N10X.Editor.AddOnInterceptKeyFunction(OnInterceptKey)

#------------------------------------------------------------------------
# the code below will be run when this script is compiled.
# Scripts are compiled on a separate thread so we need to schedule the
# Initialise in the main thread update.
if QuickPaneEnabled:
	N10X.Editor.AddUpdateFunction(Initialise)
