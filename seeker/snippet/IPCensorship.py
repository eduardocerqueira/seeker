#date: 2021-09-13T17:22:21Z
#url: https://api.github.com/gists/41d205e75b793ee9a5de52e764e08bc5
#owner: https://api.github.com/users/FolfyBlue

import sys
import os
import re
import gzip

### ARGUMENTS ###

if not len(sys.argv) == 2: # No arguments given
	print("""
help: python IPCensorship.py [DIRECTORY] <Replacement String>

Options:
    Replacement String	String with which to replace the IPs. Defaults to an empty string.

Arguments:
    DIRECTORY	Directory of a "logs" folder.
""")
	sys.exit() # Exit program

directory = sys.argv[1]

if len(sys.argv)>2: # If a replacement string was specified
	replacement = ''.join(sys.argv[2:]) # Join the rest of the "arguments" into one string as intended
else:
	replacement = "" # No replacement was given, leave empty

print('Using replacement: "'+replacement+'"')

if not os.path.exists(directory) or not os.path.isdir(directory): # Checks if we were given a valid directory (Exists, is a directory)
	print("Invalid directory! ('"+dir+"')")
	sys.exit() # Exit program

### FUNCTIONS ###

def rem_ips(string: str) -> str: # Remove IPs from a string
	string,x = re.subn("/(\d+\.){3}\d+","/"+replacement,string) #Matches strings following the format /x.x.x.x and removes them
	string,y = re.subn(":(\d+\.){3}\d+",":"+replacement,string) #Matches strings following the format :x.x.x.x and removes them
	return string,x+y

### LOGIC ###

for file_name in os.listdir(directory): # List all the files in the directory
    file_name = os.path.join(directory, file_name) # Get the full file name
    if os.path.isfile(file_name): # If the listed element ACTUALLY is a file...
    	file_name, file_extension = os.path.splitext(file_name) # Split the file name and the extension
    	if file_extension == ".log": # If it's a simple .log file...
    		with open(file_name+file_extension,"r") as log_file: # Read the file content and store it to a variable
    			file_content = log_file.read()

    		file_content,operations = rem_ips(file_content) # Remove the IPs from the string containing the file's content
    		
    		with open(file_name+file_extension,"w") as log_file: # Overwrite the file with the string containing the edited file content
    			log_file.write(file_content)
    		print("Removed "+str(operations)+" IPs from log file '"+file_name+file_extension+"'")

    	elif file_extension == ".gz": #If the log file is compressed with gzip...
    		with gzip.open(file_name+file_extension,"r") as decompressed_log_file: # Open the file in read mode with gzip, which lets us read compressed content
    			file_content = decompressed_log_file.read() # Gzip returns the file content as bytes.
    		
    		file_content = file_content.decode("utf-8") # Convert the bytes object to a string we can work with.
    		file_content,operations = rem_ips(file_content) # Remove the IPs from the string containing the file's content

    		with gzip.open(file_name+file_extension,"wb") as uncompressed_log_file: # Open the file with gzip again, but this time in "wb" mode, which means "write bytes"
    			uncompressed_log_file.write(file_content.encode()) # We overwrite the file with the edited string, which we encode back to bytes then compress to gzip
    		print("Removed "+str(operations)+" IPs from compressed log file '"+file_name+"'")
    	else:
    		print("Unsupported file type: "+file_extension)
    else:
    	print("Warning! '"+file_name+"' is not a file!")
