#date: 2022-12-22T16:59:41Z
#url: https://api.github.com/gists/a4cf53375cfffb156e818fb161580bd5
#owner: https://api.github.com/users/CharlesChiuGit

import os
import re

from delete_unused_assets import get_all_assets


def get_all_referenced_image_filenames() -> list[tuple[str, str]]:
    current_dir = os.getcwd()

    journals_dir = os.path.join(current_dir, 'journals')
    pages_dir = os.path.join(current_dir, 'pages')

    journals_files = ['journals/'+f for f in os.listdir(journals_dir) if f.endswith('.md')]
    pages_files = ['pages/'+f for f in os.listdir(pages_dir) if f.endswith('.md')]
    all_files = journals_files + pages_files

    image_references = []

    for filename in all_files:
        with open(filename, 'r') as f:
            file_contents = f.read()
        

        for item in re.findall(r'!\[\]\((.+?)\)', file_contents):
            image_references.append((filename, item))
    image_references = [(item[0], item[1][10:]) for item in image_references if item[1].startswith("../")]
    return image_references


if __name__=='__main__':
    all_referenced = get_all_referenced_image_filenames()
    all_assets = get_all_assets()
    for md_filename, image_filename in all_referenced:
        if image_filename not in all_assets:
            print(f"Image {image_filename} in {md_filename} missing")