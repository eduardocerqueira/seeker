#date: 2021-11-12T17:10:35Z
#url: https://api.github.com/gists/02ff9a47335bb17a5175544676ff32f5
#owner: https://api.github.com/users/MiCurry

import sys
import argparse

from netCDF4 import Dataset

description = ("Trasnlate the 'ter' variable of an MPAS static file to be in surface geopotential,"\
               " and rename the variable to be PHIS.")

gravity = 9.80616

parser = argparse.ArgumentParser(description=description)
parser.add_argument('file',
                    help='MPAS Static filename',
                    type=str)

args = parser.parse_args()

mesh = Dataset(args.file, 'r+')

if 'ter' not in mesh.variables:
    print("ERROR: ter not in the given file")
    sys.exit(-1)

ter = mesh.variables['ter'][:]
print(mesh.variables['ter'])
print("Converting 'ter' to geopotential using gravity as", gravity, "...")
PHIS = ter[:] * gravity

print("Renaming 'ter' to be 'PHIS', changing PHIS long_name and units and writing PHIS..")
mesh.renameVariable('ter','PHIS')
mesh.variables['PHIS'].long_name = "Surface geopotential"
mesh.variables['PHIS'].units = "m2/s2"
mesh.variables['PHIS'][:] = PHIS
mesh.close()
print("Finished!")