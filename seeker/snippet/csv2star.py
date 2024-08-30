#date: 2024-08-30T16:56:05Z
#url: https://api.github.com/gists/7781ef8063e0102d7121d799efcb700f
#owner: https://api.github.com/users/shahpnmlab

import os
import glob
import pandas as pd
import starfile

# Function to read CSV files
def read_csv_files(directory):
    all_data = []
    for filename in glob.glob(os.path.join(directory, '*.csv')):
        df = pd.read_csv(filename, header=None, names=['X', 'Y', 'Z'])
        tomogram_name = os.path.basename(filename).split('_')[1]
        df['MicrographName'] = f'TS_{tomogram_name}'
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Read all CSV files
data = read_csv_files('particle_lists')

# Create the particles data block
particles_data = pd.DataFrame({
    'rlnMicrographName': data['MicrographName'],
    'rlnCoordinateX': data['X'],
    'rlnCoordinateY': data['Y'],
    'rlnCoordinateZ': data['Z'],
    'rlnOriginXAngst': [0] * len(data),
    'rlnOriginYAngst': [0] * len(data),
    'rlnOriginZAngst': [0] * len(data)
})

# Create the optics data block
optics_data = pd.DataFrame({
    'rlnOpticsGroup': [1,""],
    'rlnOpticsGroupName': ['opticsGroup1',""],
    'rlnSphericalAberration': [2.700000,""],
    'rlnVoltage': [300.000000,""],
    'rlnImagePixelSize': [13.48,""],
    'rlnImageSize': [64,""],
    'rlnImageDimensionality': [3,""],
    'rlnPickingImagePixelSize': [13.48,""]
})

# Create the STAR file structure
star_data = {
    'optics': optics_data,
    'particles': particles_data
}

# Write the STAR file
starfile.write(star_data, 'particles.star', overwrite=True)

print("particles.star file has been created successfully.")