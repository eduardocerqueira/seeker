#date: 2025-06-30T17:11:02Z
#url: https://api.github.com/gists/42913067c14e53f1d8ffbde1bf91fcca
#owner: https://api.github.com/users/proteus1121

import os
import requests
from zipfile import ZipFile
import base64
import time

def download_image(url, folder):
    filename = os.path.join(folder, url.split('/')[-1])
    tries = 6
    for attempt in range(tries):
        try:
            with requests.get(url, stream=True, timeout=10) as r:
                r.raise_for_status()
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            return f"✔ Downloaded: {filename}"
        except requests.exceptions.Timeout:
            print(f"⚠ Timeout downloading {url}, attempt {attempt+1}/{tries}")
            time.sleep(2)
        except Exception as e:
            return f"❌ Failed {url}: {e}"
    return f"❌ Failed {url} after {tries} attempts"

def download_all_images(folder, urls):
    print("📥 Downloading images sequentially...")
    os.makedirs(folder, exist_ok=True)
    for url in urls:
        print(f"📥 Downloading image: {url}")
        result = download_image(url, folder)
        print(result)

def zip_folder(folder, zip_filename):
    print(f"📦 Creating zip archive: {zip_filename}")
    with ZipFile(zip_filename, 'w') as zipf:
        for file in os.listdir(folder):
            zipf.write(os.path.join(folder, file), arcname=file)

def save_zip_as_base64(zip_path, output_txt):
    with open(zip_path, "rb") as zip_file:
        encoded = base64.b64encode(zip_file.read()).decode('utf-8')
    with open(output_txt, "w") as txt_file:
        txt_file.write(encoded)
    print(f"📄 Base64 ZIP saved as: {output_txt}")

def main():
    folder = "sinners-reward-004"
    zip_filename = "sinners-reward-004.zip"
    base_url = "https://img.drawnstories.ru/img/IDW-comics/silent-hill/sinners-reward/sinners-reward-004/"
    image_urls = [f"{base_url}{str(i).zfill(3)}.jpg" for i in range(0, 24)]
    
    download_all_images(folder, image_urls)
    zip_folder(folder, zip_filename)
    save_zip_as_base64(zip_filename, "sinners-reward-004.txt")

if __name__ == "__main__":
    main()
