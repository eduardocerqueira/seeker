#date: 2022-03-22T17:06:40Z
#url: https://api.github.com/gists/64b594c3c38c52e5c25c95866ca9ec26
#owner: https://api.github.com/users/Lubba-64

import os
import shutil
to_clean = './compressed'
output = './full_cleaned'
if not os.path.exists(output):
    os.makedirs(output)

def clean(dir:str):
    children = [f'{dir}/{child}' for child in os.listdir(dir)]
    files = [child for child in children if os.path.isfile(child)]
    for file in files:
        file_base = os.path.basename(file)
        print(file_base, )
        if file_base.startswith('IMG_'):
            shutil.copy(file,f'{output}/{file_base}')
            print(f'copy:{file_base}')

clean(to_clean)
print('done!')