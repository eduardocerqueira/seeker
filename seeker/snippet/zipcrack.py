#date: 2023-02-08T17:09:28Z
#url: https://api.github.com/gists/dad59bc4fbb07532a7d24724b1a72948
#owner: https://api.github.com/users/ivythornb

import zipfile
import itertools
import time

# Function for extracting zip files to test if the password works!
 "**********"d "**********"e "**********"f "**********"  "**********"e "**********"x "**********"t "**********"r "**********"a "**********"c "**********"t "**********"F "**********"i "**********"l "**********"e "**********"( "**********"z "**********"i "**********"p "**********"_ "**********"f "**********"i "**********"l "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
    try:
        zip_file.extractall(pwd= "**********"
        return True
    except KeyboardInterrupt:
        exit(0)
    except Exception:
        pass

# Main code starts here...
# The file name of the zip file.
zipfilename = 'data.zip'
# The first part of the password. We know this for sure!
first_half_password = "**********"
# We don't know what characters they add afterwards...
# This is case sensitive!
alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
zip_file = zipfile.ZipFile(zipfilename)

# We know they always have 3 characters after Super...
# For every possible combination of 3 letters from alphabet...
for c in itertools.product(alphabet, repeat=3):
    # Slowing it down on purpose to make it work better with the web terminal
    # Remove at your peril
    time.sleep(0.009)
    # Add the three letters to the first half of the password.
    password = "**********"
    # Try to extract the file.
    print ("Trying: "**********"
    # If the file was extracted, you found the right password.
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"e "**********"x "**********"t "**********"r "**********"a "**********"c "**********"t "**********"F "**********"i "**********"l "**********"e "**********"( "**********"z "**********"i "**********"p "**********"_ "**********"f "**********"i "**********"l "**********"e "**********", "**********"  "**********"s "**********"t "**********"r "**********". "**********"e "**********"n "**********"c "**********"o "**********"d "**********"e "**********"( "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********") "**********": "**********"
        print ('*' * 20)
        print ('Password found: "**********"
        print ('Files extracted...')
        exit(0)

# If no password was found by the end, let us know!
print ('Password not found.')