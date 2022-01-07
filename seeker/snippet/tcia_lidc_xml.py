#date: 2022-01-07T17:03:28Z
#url: https://api.github.com/gists/7b75ed38fb5665f577b644661b54cbb4
#owner: https://api.github.com/users/taznux

import glob
import re
import shutil
import pandas as pd
import xml.etree.ElementTree as ET

path_lidc = "manifest-1600709154662/LIDC-IDRI"

meta = pd.read_csv("LIDC-IDRI_MetaData.csv")
print(meta)
xml_files = glob.glob("tcia-lidc-xml/*/*.xml")
print(len(xml_files))
for xml_file in xml_files:
    with open(xml_file) as f:
        xmlstring = f.read()

        # Remove the default namespace definition (xmlns="http://some/namespace")
        xmlstring = re.sub(' xmlns="[^"]+"', '', xmlstring, count=1)
        #print(xmlstring)
        root = ET.fromstring(xmlstring)
        rh = root.find("ResponseHeader")
        try:
            seriesUID = rh.find("SeriesInstanceUid").text
        except:
            seriesUID = rh.find("SeriesInstanceUID").text
        studyUID = rh.find("StudyInstanceUID").text
        subjectID = meta[meta["Study UID"].str.match(studyUID)]["Subject ID"].array[0]
        dest = f"{path_lidc}/{subjectID}/{studyUID}/{seriesUID}/"

        print(xml_file, dest)
        try:
            shutil.copy(xml_file, dest)
            print("File copied successfully.")

        # If source and destination are same
        except shutil.SameFileError:
            print("Source and destination represents the same file.")

        # If destination is a directory.
        except IsADirectoryError:
            print("Destination is a directory.")

        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")

        # For other errors
        except:
            print("Error occurred while copying file.")
