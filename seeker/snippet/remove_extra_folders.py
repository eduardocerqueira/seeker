#date: 2022-03-22T17:06:40Z
#url: https://api.github.com/gists/64b594c3c38c52e5c25c95866ca9ec26
#owner: https://api.github.com/users/Lubba-64

import os
import shutil

output = './compressed'
if not os.path.exists(output):
    os.makedirs(output)

to_clean = './Masters'

current_root = ''

def clean(dir:str):
    children = [f'{dir}/{child}' for child in os.listdir(dir)]
    
    folders = [child for child in children if os.path.isdir(child)]
    files = [child for child in children if os.path.isfile(child)]

    for folder in folders:
        clean(folder)
    
    for file in files:
        file_base = os.path.basename(file)
        shutil.copy(file,f'{output}/{file_base}')
        print(f'copy:{file_base}')

clean(to_clean)
print('done!')