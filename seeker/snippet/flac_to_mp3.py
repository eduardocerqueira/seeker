#date: 2023-11-29T16:48:25Z
#url: https://api.github.com/gists/2f093aa49f39bf20addd91c327e7d97a
#owner: https://api.github.com/users/bitsydoge

#!/usr/bin/env python

# Author: bitsydoge
# Description: This script converts FLAC files to MP3 format using ffmpeg-python
# Usage: flac_to_mp3.py [-h] [--input_folder INPUT_FOLDER] [--output_folder OUTPUT_FOLDER] [--bitrate BITRATE] [--verbose]

import os
import ffmpeg
import argparse
import tqdm

# Command-line arguments setup using argparse
parser = argparse.ArgumentParser(description='Convert FLAC files to MP3')
parser.add_argument('--input_folder', default=".", help='Path to the input folder containing FLAC files')
parser.add_argument('--output_folder', default="mp3_output", help='Path to the output folder for converted MP3 files')
parser.add_argument('--bitrate', type=int, default=320000, help='Bitrate for the output MP3 files')
parser.add_argument('--verbose', action='store_true', help='Show verbose of ffmpeg')
args = parser.parse_args()

# Create the output folder if it doesn't exist
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

flac_files = [f for f in os.listdir(args.input_folder) if f.endswith('.flac')]

files_converted = 0

for file in tqdm.tqdm(flac_files, desc="Converting", unit="file"):
    input_file = os.path.join(args.input_folder, file)
    output_file = os.path.join(args.output_folder, os.path.splitext(file)[0] + '.mp3')

    # Convert FLAC to MP3 using ffmpeg
    ffmpeg.input(input_file).output(output_file, bitrate=args.bitrate, loglevel="info" if args.verbose else "quiet").run(overwrite_output=True)

    files_converted += 1

print(f"Conversion complete. {files_converted} files converted.")