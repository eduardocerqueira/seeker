#date: 2021-10-25T17:09:23Z
#url: https://api.github.com/gists/0fcae87071ef4b448d1b8c0f89628e33
#owner: https://api.github.com/users/brandondube

from io import BytesIO

import requests

from imageio import imread

if __name__ == '__main__':
    url = 'http://localhost:5000/wrong-gpu-and-memory-leak'
    while True:
        resp = requests.get(url)
        b = BytesIO(resp.content)
        b.seek(0)
        im = imread(b, format='tiff')
