#date: 2023-04-13T17:03:52Z
#url: https://api.github.com/gists/12e7e5c4417049af22fbe7fcf8f01134
#owner: https://api.github.com/users/naderidev

import m3u8
import requests

def local_write(file_path: str, segment: m3u8.Segment, m3u8_base_url: str, mode: str = '+ab'):
    with open(file_path, mode) as file:
        req = requests.get(m3u8_base_url + segment.uri)
        if req.status_code == 200:
            file.write(req.content)