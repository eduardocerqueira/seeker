#date: 2022-05-05T16:52:16Z
#url: https://api.github.com/gists/ed6dfa39a433b2f517b19fd4c5a81c4a
#owner: https://api.github.com/users/rileyhales

# requires: pandas, nco
# conda install -c conda-forge nco pandas
# python >=3.10

import os
import datetime
import subprocess
import pandas as pd
import sys

if __name__ == "__main__":
    year = int(sys.argv[1][:4])
    daterange = pd.date_range(
        datetime.date(year, 1, 1),
        datetime.date(year, 12, 31),
        freq="D"
    )
    for date in daterange:
        print(date)
        wildcard_path = os.path.join(".",f"{ year}", f"{date.strftime('%Y%m%d')}*")
        print(wildcard_path)
        subprocess.call(f'nces {wildcard_path} -o {date.strftime("%Y%m%d")}.nc --op_typ mean', shell=True)

