#date: 2024-12-20T16:48:38Z
#url: https://api.github.com/gists/9a5eb2b67a78c5b9e1dd0a6467e45bea
#owner: https://api.github.com/users/ABHIRAMSHIBU

#!/usr/bin/env python3
# Author: Abhiram Shibu <abhiramshibu1998@gmail.com>

import os
import shutil
import subprocess

def merge_video_parts(group_files, output_file):
    """Merge video parts using ffmpeg concat demuxer."""
    # Create temporary file list
    with open('file_list.txt', 'w') as f:
        for part in sorted(group_files):
            f.write(f"file '{part}'\n")
    
    # Merge parts using ffmpeg
    cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', 
           '-i', 'file_list.txt', '-c', 'copy', output_file]
    subprocess.run(cmd)
    
    # Clean up temporary file
    os.remove('file_list.txt')

def main():
    # Configuration
    OUTPUT_DIR = 'merge'
    
    # Prepare output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    # Collect and categorize files
    part_files = []
    non_part_files = []
    for file in os.listdir():
        if file.endswith('.mp4'):
            if '_part' in file:
                part_files.append(file)
            else:
                non_part_files.append(file)
    
    if not part_files:
        print("No video parts found")
        return -1
    
    # Group files by base name
    file_groups = {}
    for part in part_files:
        base_name = part.split('_part')[0]
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(part)
    
    # Process each group
    for base_name, group_files in file_groups.items():
        output_file = os.path.join(OUTPUT_DIR, f'{base_name}.mp4')
        merge_video_parts(group_files, output_file)
    
    # Create symlinks for non-part files
    for file in non_part_files:
        source = os.path.abspath(file)
        destination = os.path.join(OUTPUT_DIR, file)
        os.symlink(source, destination)

if __name__ == '__main__':
    main()
