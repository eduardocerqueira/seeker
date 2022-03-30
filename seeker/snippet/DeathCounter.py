#date: 2022-03-30T16:52:39Z
#url: https://api.github.com/gists/cd1fb2cbe42f23ab9bb81225801fa124
#owner: https://api.github.com/users/Riffecs

# Death Counter - Elden Ring
import os
import keyboard
import sys

# Load Global Vars
# Field to increase counter.
counter = 'e'
# Field to cancel the programme
end = 'q'
# Path to the logger file
# Please enter the path with double backslash
path = "C:\\Users\\Riffecs\\Desktop\\Leon\\counter.txt"
# If no file is available, this is taken as the start value.
start = 30

# Splitting the path into different elements    
pathsplit = path.split("\\")

# Testing the file name
filename = pathsplit[len(pathsplit)-1]

# Compose the new path
pathnew = ""

# It is recommended that the file is not included here. Therefore, the for-each is modified.
for i in range(0, len(pathsplit) - 1):
    pathnew += pathsplit[i]+"\\"

# Change path
try:
    os.chdir(pathnew)
except Exception as e:
    print("The path is not available. Please check if you have made a mistake")
    print("Exception: "+e)
    sys.exit(0)
finally:
    print("Current folder: "+os.getcwd())

# Test if the file already exists
# If the file does not exist, it is created.
try:
    f = open(pathsplit[len(pathsplit)-1], encoding="utf-8")
except IOError as file_error:
    print("File not accessible")
    print("I am trying to create the file.")
    f = open(pathsplit[len(pathsplit)-1], "w+", encoding="utf-8")
    f.write(str(start))
finally:
    f.close()

# This is where the actual programme begins.
# The rest was actually just configuration stuff.


# Basically, it works like a Mugen Tsukuyomi. 
# That's why it's tied into an endless Loop. 
while True: 
    # Here is when the button is pressed.
    if keyboard.read_key() == counter:
        # Creating the File Handler
        f =  open(pathsplit[len(pathsplit)-1], "r+", encoding="utf-8")
        runner = f.readline()
        runner = runner.replace(" ", "")
        runner = runner.replace("\x00","")  
        # Outpute
        print("Runner:"+ runner)
        print("Type:" + str(type(runner)))
        runner = int(runner)
        # Reading the counter
        runner += 1
        # Emptying the file at byte level
        f.seek(0)
        # Describe the file with the new countere
        f.write(str(runner))
        # Exit the File Handlere
        f.close()

    # End of the Tsukuyomi
    if keyboard.read_key() == end:
        print("Programm end")
        break
        
# Exit the complete programme. 
sys.exit(0)