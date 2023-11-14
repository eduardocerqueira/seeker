#date: 2023-11-14T16:52:00Z
#url: https://api.github.com/gists/a51b25a1be7dcf73387881552452bfd0
#owner: https://api.github.com/users/InfraRedApple

import os, hashlib, sys, codecs
from os.path import join, getsize

# This Python script is designed to search a specified directory and its subdirectories for duplicate files, 
# based on both their file size and the SHA-512 hash of their contents. 

# Define the directory to search for duplicate files
DIRECTORY = '/full/path/to/directory'

# Define the buffer size for reading files in chunks
BUFFERSIZE = 64 * 1024

# Initialize a counter for deleted files
deleteCount = 0

# Function to calculate the SHA-512 hash of a file
def hash_file(filename):
    try:
        # Open the file in binary read mode
        file = open(filename, 'rb')
        hashed = hashlib.sha512()
        
        # Read the file in chunks and update the hash
        while True:
            buff = file.read(BUFFERSIZE)
            if not buff:
                break
            hashed.update(buff)
        
        # Return the SHA-512 hash value as a byte string
        return hashed.digest()
    except IOError as err:
        # Print an error message if the file cannot be read
        print(err)

# Create a dictionary to group files by their size
samesize = {}

# Traverse the directory and its subdirectories
for path, _, filenames in os.walk(DIRECTORY):
    for filename in filenames:
        fullname = os.path.join(path, filename)
        
        # Group files by their size
        samesize.setdefault(os.path.getsize(fullname), []).append(fullname)

# Print the total number of unique files to process
print('Total Unique Files To Process: ' + str(len(samesize)))

# Loop through files with the same size
for filenames in samesize.values():
    if len(filenames) > 1:
        # Create a dictionary to group files by their SHA-512 hash
        hashes = {}
        
        # Calculate the SHA-512 hash for each file in the group
        for filename in filenames:
            hashes.setdefault(hash_file(filename), []).append(filename)

        # Print and delete duplicate files with the same hash
        for identicalfiles in hashes.values():
            if len(identicalfiles) > 1:
                print('Delete: ' + identicalfiles[1] + '\n')
                os.remove(identicalfiles[1])
                deleteCount += 1

# Print the total number of deleted files
print('FINISHED -- Deleted: ' + str(deleteCount))

