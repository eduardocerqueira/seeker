#date: 2023-06-21T16:41:18Z
#url: https://api.github.com/gists/6cdebcafc84408c3c10d58c07c1fbcb6
#owner: https://api.github.com/users/VoidAny

#!/usr/bin/env python3

"""
This script will add all the torrents in the BT_backup folder to transmission
It will also add the labels and download directory from the corresponding quickresume file
Once a torrent is added, it will be renamed to .added so it won't be added again (if the script is run again)

Make sure to edit the Client() settings to match your transmission settings and the path to the BT_backup folder

Run:
pip3 install transmission-rpc bencodepy 
before running this script
"""
from transmission_rpc import Client, Torrent
from transmission_rpc.error import TransmissionTimeoutError
import time
import bencodepy
from pathlib import Path

# Edit here to match your transmission settings
client = "**********"='http', host='127.0.0.1', port=9091, path="/transmission/rpc", username="", password="")

# Replace the path to the path to your BT_backup folder
torrent_files = Path('BT_backup').glob('*.torrent')

def get_labels(torrent: Path):
    """Open the corresponding quickresume file in the BT_backup folder and return the tags"""
    with torrent.with_suffix('.fastresume').open('rb') as f:
        fastresume = bencodepy.decode(f.read())
    labels = [fastresume[b'qBt-category'].decode()] + [t.decode() for t in fastresume[b'qBt-tags']]
    # You can add more if statements here to change the labels to whatever you want
    if 'sonarr' in labels:
        labels.remove('sonarr')
        labels.append("TV/Anime")
    if "radarr" in labels:
        labels.remove('radarr')
        labels.append("Movies")

    return labels

def get_download_dir(torrent: Path):
    """Open the corresponding quickresume file in the BT_backup folder and return the download directory"""
    with torrent.with_suffix('.fastresume').open('rb') as f:
        fastresume = bencodepy.decode(f.read())
    return fastresume[b'save_path'].decode()


for torrent_file in torrent_files:
    labels = get_labels(torrent_file)
    try:
        torrent: Torrent = client.add_torrent(torrent_file, download_dir=get_download_dir(torrent_file), labels=labels)
    except TransmissionTimeoutError:
        print("Error: Transmission timed out")
        print("Trying again in 30 seconds")
        time.sleep(30)
        torrent = client.add_torrent(torrent_file, download_dir=get_download_dir(torrent_file), labels=labels)

    print(f"Added {torrent.name} to transmission (labels: {','.join(labels)})")
    # You can adjust this sleep time if you want
    time.sleep(7)
    torrent_file.rename(torrent_file.with_suffix('.added'))

)

