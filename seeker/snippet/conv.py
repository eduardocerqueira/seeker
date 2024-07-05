#date: 2024-07-05T16:43:58Z
#url: https://api.github.com/gists/4c5ec98b00f68a979aba1f5ab85c7d69
#owner: https://api.github.com/users/BitesizedLion

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

directory_path = './'
acrcloud_path = '/tmp/tmp.KINBD0Xo6M/acrcloud'
output_base_path = './'

video_extensions = ['.mkv', '.mp4']

def process_file(file_path):
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_base_path, f"{file_name}.db.lo")
    command = [acrcloud_path, '-i', file_path, '-o', output_path]
    try:
        print(f"Processing... {file_name}")
        subprocess.run(command, check=True)
        print(f"Processed {file_path} successfully. Output: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to process {file_path}. Error: {e}")

video_files = []
for root, _, files in os.walk(directory_path):
    for file in files:
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(root, file))

with ThreadPoolExecutor() as executor:
    executor.map(process_file, video_files)