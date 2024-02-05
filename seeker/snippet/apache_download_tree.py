#date: 2024-02-05T17:08:59Z
#url: https://api.github.com/gists/70ed5df26e929af122aab3059f9aa5c6
#owner: https://api.github.com/users/fjourdren

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from collections import deque

def should_skip_link(href):
    # Define all conditions to skip a link
    return href in {'../', '/', '?C=N;O=D', '?C=M;O=A', '?C=S;O=A', '?C=D;O=A', '', '#'} or href.startswith('..') or href.startswith('/')

def download_files(url, local_path='.'):
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    
    queue = deque([(url, local_path)])
    
    while queue:
        current_url, current_path = queue.popleft()
        response = requests.get(current_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for link in soup.find_all('a'):
            href = link.get('href')
            if should_skip_link(href):
                continue  # Skip links based on the specified conditions

            next_url = urljoin(current_url, href)
            next_path = os.path.join(current_path, href.rstrip('/'))
            
            if href.endswith('/'):
                # It's a directory
                if not os.path.exists(next_path):
                    os.makedirs(next_path)
                queue.append((next_url, next_path))
            else:
                # It's a file, check if it already exists before downloading
                if not os.path.exists(next_path):
                    print(f"Downloading {next_url} to {next_path}")
                    download_file(next_url, next_path)
                else:
                    print(f"File {next_path} already exists. Skipping download.")

def download_file(url, path):
    with requests.get(url, stream=True) as r, open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# Usage example
url = 'https://apache_url/medias/'
download_files(url, 'download')
print('Download complete!')