#date: 2021-12-01T17:05:49Z
#url: https://api.github.com/gists/79b19f5788d3d9136a264405f868f05c
#owner: https://api.github.com/users/BlueDev1

import os
i = 0
def start():
    check = input('Welcome! Press any key to continue. \n')
    print('1. for directory list \n ')
    print('2. for rename \n ')
    print('3. to quit \n')


start()
menu = int(input('Select one of the options you want me to do. \n'))


if menu == 1:
    dirpath = input('Enter the path to the directory you want a listing of. \n')
    dirlist = os.listdir(dirpath)
    print(dirlist)


if menu == 2:
    input('Put all the files you want to rename in the directory that this script is in. Once you are done press any key to continue. \n')
    file_name = input('Enter the new name for the files. \n')
    file_extension = input('Enter the extension you want for the files. \n')
    for file in os.listdir():
      src=file
      dst=file_name+str(i)+ file_extension
      if file == 'autorenamer.py':
          continue
      os.rename(src,dst)
      i+=1

if menu == 3:
    quit()

