#date: 2022-12-22T16:59:41Z
#url: https://api.github.com/gists/a4cf53375cfffb156e818fb161580bd5
#owner: https://api.github.com/users/CharlesChiuGit

import os
import shutil


assets_dir = './assets'
journal_dir = './journals'
pages_dir = './pages'
to_delete_dir = './to_delete'

def get_all_assets() -> list[str]:
    if not os.path.exists(to_delete_dir):
        os.makedirs(to_delete_dir)

    assets_files = os.listdir(assets_dir)
    return assets_files


if __name__=='__main__':
    assets_files = get_all_assets()
    referenced_files = []


    for dirname in [journal_dir, pages_dir]:
        for filename in os.listdir(dirname):
            if filename.endswith('.md'):
                with open(os.path.join(dirname, filename)) as f:
                    for line in f:
                        for asset in assets_files:
                            if asset in line:
                                referenced_files.append(asset)



    for asset in assets_files:
        if asset not in referenced_files and not asset.endswith(".edn"):
            print("assets/"+asset)
            shutil.move(os.path.join(assets_dir, asset), to_delete_dir)