#date: 2024-09-19T16:35:55Z
#url: https://api.github.com/gists/d11e9a90eff14b7eedb0654a42a485cb
#owner: https://api.github.com/users/zonk-labs

import re
import requests
import os
from requests.auth import HTTPBasicAuth
from concurrent.futures import ThreadPoolExecutor

baseurl = 'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/'
session = requests.Session()

 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"f "**********"r "**********"o "**********"m "**********"_ "**********"e "**********"n "**********"v "**********"( "**********") "**********": "**********"
    earthdata_token = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"e "**********"a "**********"r "**********"t "**********"h "**********"d "**********"a "**********"t "**********"a "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
        print("You need to create an API key on urs.earthdata.nasa.gov and plug it into the EARTHDATA_TOKEN environment variable.")
        exit()
    else:
        return {'Authorization': "**********"

def write_index(filelist):
    try:
        with open('index.idx', 'w', encoding='utf-8', newline='') as file:
            for item in filelist:
                file.write(str(item) + '\n')
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

def load_index():
    try:
        with open('index.idx', 'r', encoding='utf-8') as file:
            filelist = [line.strip() for line in file]
            return filelist

    except IOError as e:
        print(f"Couldn't read file: {e}")
        return []

def get_completed():
    dircontents = os.listdir()
    return [zipped for zipped in dircontents if os.path.isfile(zipped) and zipped.endswith('.zip')]

done = get_completed()

def get_filelist():
    if os.path.exists('index.idx'):
        return load_index()

    req = session.get(baseurl, auth=basic)
    regexp = re.compile(r'<a href=\"(ASTGTMV003_\S*\.zip)\">\S*<\/a>')
    matches = regexp.findall(req.text)
    filelist = [match[0] if isinstance(match, tuple) else match for match in matches]
    write_index(filelist)

    return filelist

tile_list = get_filelist()

def download_file(file):
    try:
        req = session.get(baseurl + file, stream=True)
        req.raise_for_status()

        with open(file, 'wb') as tile:
            for chunk in req.iter_content(chunk_size=8192):
                tile.write(chunk)

        done.append(file)
        print(f"{file} - {len(done)/len(tile_list) * 100:.2f}% ({len(done)} of {len(tile_list)})")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading tile: {e}")
    except IOError as e:
        print(f"Couldn't write tile data: {e}")
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    session.headers.update(get_token_from_env())

    done = get_completed()
    percent_done = len(done) / len(tile_list)

    print(f"{percent_done * 100:.2f} % done ({len(done)}/{len(tile_list)})")

    with ThreadPoolExecutor(max_workers=10) as executor:
        try:
            futures = []
            for tile in tile_list:
                if tile not in done:
                    futures.append(executor.submit(download_file, tile))
            executor.shutdown(wait=True)
        except KeyboardInterrupt:
            print("Shutting down...")
            executor.shutdown(cancel_futures=True)
        finally:
            print("Done.")