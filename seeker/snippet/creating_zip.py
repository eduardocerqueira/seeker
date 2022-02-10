#date: 2022-02-10T16:56:47Z
#url: https://api.github.com/gists/eb27541071ebb3abdfd743d8c903800e
#owner: https://api.github.com/users/sunilkumarvalmiki

# 1 | move all the files to a directory, where the code " creating_zip.py " file is present.
# 2 | download these file and make the necessary change. 
  # a | <name> - name of the zip file, user choice 
  # b | <file-name> - name of the file, which needs to be zipped.
  
# importing zipfile module | in - built module
import zipfile 

# creating a zipfile with already existing files in the directory.
with zipfile.ZipFile('<name>', mode="w") as myZip:
    myZip.write('<file-name>')
    myZip.write('<file-name>')