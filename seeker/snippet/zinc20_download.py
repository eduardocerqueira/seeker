#date: 2023-02-24T16:58:46Z
#url: https://api.github.com/gists/70751345ec55893cf6cffffa8591ca35
#owner: https://api.github.com/users/MartinEls

import requests
from tqdm import tqdm

def download(zinc20_url) 
    tranch = (zinc20_url.split('/')[-1]).split('.')[0]
    r = requests.get(zinc20_url.strip())
    with open('all_tranches/' + tranch + '.tsv', 'wb') as f:
        f.write(r.content)
        
if __name__ == '__main__':
    with open('ZINC-downloader-2D-txt.uri', 'r') as f:
    urls = f.readlines()
    for url in tqdm(urls):
        try:
            download(url)
        except:
            print(f'Downloading from {url} failed')