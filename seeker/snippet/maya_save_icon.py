#date: 2023-07-07T16:55:13Z
#url: https://api.github.com/gists/94fc126c8c48a22e429301aa75c9e638
#owner: https://api.github.com/users/L0Lock

from pymel.core import *
import os
import maya.api.OpenMaya as om
def main():
    ###### USER CUSTOMIZATION ######
    ICONS_ITEMS = "play_regularrrrr.png"
    OUTPUT_DIRECTORY = "C:/temp/icons/"
    # Don't forget the closing slash!
    ######   ######  ######   ######

############################################################################
    # Creates the folder if doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    try:
        for item in resourceManager(nameFilter=ICONS_ITEMS):
            try:
                resourceManager(saveAs=(item, '{}{}'.format(OUTPUT_DIRECTORY,item)))
                print("Saved : {}".format(item))
            except:
                #For the cases in which some files do not work for windows, name formatting wise. I'm looking at you 'http:'!
                print("Couldn't save : {}".OUTPUT_DIRECTORY+item)

        om.MGlobal.displayInfo("Done saving icons. See script editor for more details")
    except:
        om.MGlobal.displayWarning("Couldn't find the icon. Check the 'ICON_ITEMS' variable content")

if __name__ == "__main__":
    main()