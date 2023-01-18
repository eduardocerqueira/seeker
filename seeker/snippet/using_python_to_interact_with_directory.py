#date: 2023-01-18T17:08:44Z
#url: https://api.github.com/gists/6054b27455d8841f37f12b0106985236
#owner: https://api.github.com/users/Malherbe

# 1
import os
def create_python_script(filename):
  comments = "# Start of a new Python program"
  with open(filename,'a') as file:
    filesize = file.write(comments)
  return(filesize)

print(create_python_script("program.py"))

#2

import os

def new_directory(directory, filename):
  # Before creating a new directory, check to see if it already exists
  if os.path.isdir(directory)==False:
    os.mkdir(directory)
  # Create the new file inside of the new directory
  path=os.path.join(directory,filename)
  # Return the list of files in the new directory
  os.mkdir(path)
  return os.listdir(directory)
  
print(new_directory("PythonPrograms", "script.py"))

#4

import os
import datetime
from datetime import date
def file_date(filename):
  # Create the file in the current directory
  os.mkdir(filename)
  timestamp = os.path.getmtime(filename)
  # Convert the timestamp into a readable format, then into a string
  timestamp=date.fromtimestamp(timestamp)
  # Return just the date portion 
  # Hint: how many characters are in “yyyy-mm-dd”? 
  return ("{}".format(timestamp))

print(file_date("newfile.txt")) 
# Should be today's date in the format of yyyy-mm-dd


#5

import os
def parent_directory():
  # Create a relative path to the parent 
  # of the current working directory 
  path = os.getcwd()
  relative_parent = os.path.abspath(os.path.join(path, os.pardir))
  # Return the absolute path of the parent directory
  return relative_parent

print(parent_directory())

