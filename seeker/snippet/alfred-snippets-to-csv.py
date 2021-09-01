#date: 2021-09-01T17:13:25Z
#url: https://api.github.com/gists/79b26505e9d0b79611dcdf308a57c83b
#owner: https://api.github.com/users/kristiewirth

import os
import pandas as pd
import json

# Update these for your use case
folder = 'FOLDER'
csv_filename = 'CSV_FILENAME'

concated_df = pd.DataFrame()

for filename in os.listdir(folder):  
    with open(f'{folder}/{filename}', 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            pass
    # Choose whether to transpose or not based on your JSON structure
    temp_df = pd.DataFrame(data).transpose()
    concated_df = pd.concat([concated_df, temp_df])

concated_df.to_csv(csv_filename, index=False)