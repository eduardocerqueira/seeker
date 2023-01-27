#date: 2023-01-27T17:09:51Z
#url: https://api.github.com/gists/285131aa867d350049a8a31e691ebf9c
#owner: https://api.github.com/users/metriics

import errno
import os
import shutil
from datetime import datetime

directory = None
destDir = None

while not directory:
    target = input("Directory to look in: ")
    if os.path.exists(target):
        directory = target
    else:
        print("{} does not exist.\n".format(target))

while not destDir:
    target = input("Directory to copy to: ")
    if os.path.exists(target):
        destDir = target
    else:
        print("{} does not exist.\n".format(target))

# https://stackoverflow.com/a/14115286
def createDir(destination):
    currentDir = os.path.join(
        destination, 
        datetime.now().strftime('%Y-%m-%d'))
    try:
        os.makedirs(currentDir)
        return currentDir
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        else:
            return currentDir # Dir exists, just add to it

dest = createDir(destDir)

print("\n{}  ->  {}".format(directory, dest))
print("Copying rated images...\n")

copied = dict()
numCopied = 0
ratingsTotal = [ 0, 0, 0, 0, 0 ]

for pic in os.listdir(directory):
    full = os.path.join(directory, pic)
    if os.path.splitext(pic)[1].upper() == ".ARW":
        if os.path.isfile(full):
            with open(full, 'rb') as imageFile: # Read XMP from file bytes https://stackoverflow.com/a/8120117
                fb = imageFile.read()
                xmp_rating_start = fb.find(b'<xmp:Rating>')
                xmp_rating_end = fb.find(b'</xmp:Rating>')
                xmp_rating = int(fb[xmp_rating_start + 12:xmp_rating_end].decode("utf-8"))
                if xmp_rating >= 1:
                    shutil.copy2(full, dest)
                    copied[pic] = xmp_rating
                    numCopied += 1
                    ratingsTotal[xmp_rating - 1] += 1

if numCopied == 0:
    print("No rated pictures found.\n")
else:
    print("Copied {} image(s) to {}".format(numCopied, dest))
    print("Stats:")
    for rating in range(0, 5):
        if ratingsTotal[rating] > 0:
            print("{stars:>5}: {num} image(s)".format(stars="*"*(rating+1), num=ratingsTotal[rating]))
    print()
                