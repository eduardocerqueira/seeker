#date: 2022-05-30T17:10:57Z
#url: https://api.github.com/gists/65ad6df23a296a2b54b0df8d3f5d5e8f
#owner: https://api.github.com/users/jemo07

# removes comments from Forth file for easy uploading.
# Jose Morales
import os
filename = input('Enter a file name: ')
#reading from files
with open(filename) as fp:
    contents=fp.readlines()
#partition the files base on the Forth comment "\"
#head, sep, tail = contents.partition('\\')
decreasing_counter=0  
for number in range(len(contents)):
    if "\\" in contents[number-decreasing_counter]:
        #Delete line starting with "\"
        if contents[number-decreasing_counter].startswith("\\"):
            contents.remove(contents[number-decreasing_counter])
            decreasing_counter+=1
        else: 
            newline=""  
            #Delete all after "\"
        for character in contents[number-decreasing_counter]:
            if character=="\\":
                contents.remove(contents[number-decreasing_counter])
                contents.insert(number-decreasing_counter,newline)
                newline+=character
    else:

# writing into a new file
        with open("new_"+ filename ,"w") as fp:
            fp.writelines(contents)