#date: 2023-10-25T16:43:33Z
#url: https://api.github.com/gists/9814fc8116b50ee477a947881c7b0bc3
#owner: https://api.github.com/users/jnhmcknight

import requests
chunk_size = 4096
filename = "logo.png"
document_url = "https://wasi0013.files.wordpress.com/2018/11/my_website_logo_half_circle_green-e1546027650125.png"
with requests.get(document_url, stream=True) as r:
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size): 
                if chunk:
                    f.write(chunk)