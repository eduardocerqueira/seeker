#date: 2024-10-28T17:10:27Z
#url: https://api.github.com/gists/92fde23a56e9787a0cf820429fbddee7
#owner: https://api.github.com/users/kwahoo2

# SoEnvironment usage in FreeCAD scene
# https://www.coin3d.org/coin/classSoEnvironment.html
# https://forum.freecad.org/viewtopic.php?p=785643#p785643
from pivy.coin import SoEnvironment
se = SoEnvironment() # create a new node
sg = Gui.ActiveDocument.ActiveView.getSceneGraph()
sg.insertChild(se, 1) # add the node at the beginning of our scene
se.ambientIntensity.getValue() # read value for current ambient light 0.2
se.ambientIntensity.setValue(0.8) # set a new value for ambient light, can be higher than 1 too, since it is multiplied by AmbientColor property of a material

# add fog
se.fogType = SoEnvironment.HAZE
# different type of fog
se.fogType = SoEnvironment.SMOKE

# make the fog red
from pivy.coin import SbColor
se.fogColor = SbColor(1, 0, 0)