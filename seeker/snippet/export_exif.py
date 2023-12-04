#date: 2023-12-04T17:05:57Z
#url: https://api.github.com/gists/6dae4fb4cfff3e4231806a1240fd5a70
#owner: https://api.github.com/users/BoMeyering

import os
import numpy as np
import pandas as pd

from exif import Image

DIRECTORY = "exif_photos" # Set top directory here

res = []

for root, dirs, files in os.walk(DIRECTORY):
	for file in files:
		with open(root + '/' + file, 'rb') as src:
			img = Image(src)
			meta_data = img.get_all()
			meta_data['filename'] = file
			del meta_data['user_comment']
			res.append(meta_data)
			print(f"File {file} exif extracted.")
exif_df = pd.DataFrame(res)

exif_df.insert(0, 'filename', exif_df.pop('filename'))

exif_df.to_csv('exifMetaData.csv')

print('EXIF extraction complete!')