#date: 2021-12-21T17:11:58Z
#url: https://api.github.com/gists/871598e16b2ab248cae563559434d1e0
#owner: https://api.github.com/users/keitazoumana

# Import Useful libraries
from urllib.request import urlretrieve
from os.path import exists
from tqdm import tqdm

def collect_data(file_url_list):
    
    # Check if the "./data" folder exists, create one otherwise
    if not exists("./data/"):
        os.mkdir("./data/")
        
    # Start the data downloading process
    for file, url in tqdm(file_url_list):
        if not os.path.exists(file):
            urlretrieve(url, file)