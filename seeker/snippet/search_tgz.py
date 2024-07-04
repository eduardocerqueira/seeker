#date: 2024-07-04T17:05:28Z
#url: https://api.github.com/gists/06135083ec2fbb9e7e5308686bbc7998
#owner: https://api.github.com/users/shoaibahmed

import re
import tarfile
import natsort
from glob import glob
from tqdm import tqdm


def search_in_tgz(tgz_path, search_pattern):
    """
    Search for files within a given tar.gz file
    """
    try:
        with tarfile.open(tgz_path, 'r:gz') as tar_ref:
            pattern = re.compile(search_pattern, re.IGNORECASE)
            for member in tar_ref.getmembers():
                if pattern.search(member.name):
                    print(f'Found {search_pattern} in {tgz_path}: {member.name}')
    except tarfile.TarError:
        print(f'Error: {tgz_path} is not a valid tar.gz file.')
    return None


if __name__ == "__main__":
    base_dir = "/mnt/sas/SAS/Google backup"
    tgz_files = list(glob(f"{base_dir}/*.tgz"))
    tgz_files = natsort.natsorted(tgz_files)
    print("TGZ files:", tgz_files[:5])
    search_pattern = r"2023/2023.*\.jpg"  # Regex pattern to search for

    for file in tqdm(tgz_files):
        search_in_tgz(file, search_pattern)
