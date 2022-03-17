#date: 2022-03-17T16:57:50Z
#url: https://api.github.com/gists/c89c1bf2064a0605873c09d3f7d48f06
#owner: https://api.github.com/users/marinavillaschi

# make sure you have installed the chardet library
# !pip install chardet

import chardet 

with open("myfile.csv", 'rb') as file:
    print(chardet.detect(file.read()))