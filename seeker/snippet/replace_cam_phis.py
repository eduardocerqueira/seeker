#date: 2021-11-12T17:10:35Z
#url: https://api.github.com/gists/02ff9a47335bb17a5175544676ff32f5
#owner: https://api.github.com/users/MiCurry

import sys
import argparse

from netCDF4 import Dataset

description = ("Given a CAM bnd_topo topography file and a corrosponding MPAS grid that contains"
               " the field 'PHIS', replace 'PHIS' in the bnd_topo file with the 'PHIS' from"
               " the MPAS static file.")


parser = argparse.ArgumentParser(description=description)
parser.add_argument('bnd_topo',
                    help='CAM Topography File',
                    type=str)
parser.add_argument('mpas_static',
                    help='MPAS file that contains the "PHIS" to replace the "PHIS" in bnd_topo',
                    type=str)

args = parser.parse_args()

bnd_topo_fname = args.bnd_topo
mpas_static_fname = args.mpas_static

bnd_topo = Dataset(bnd_topo_fname,'r+')
mpas_static = Dataset(mpas_static_fname,'r+')

if 'PHIS' not in mpas_static.variables:
    print("ERROR: mpas_static file does not contain a 'PHIS' variable")
    sys.exit(-1)


ncells = mpas_static.dimensions['nCells'].size
ncol = bnd_topo.dimensions['ncol'].size

if ncells != ncol:
    print("ERROR: ncol {0} from {1} and nCells {2} from {3} are not the same".format(ncol,
                                                                                     bnd_topo_fname,
                                                                                     ncells,
                                                                                     mpas_static_fname))
    sys.exit(-1)

print("Replacing 'PHIS' in {0} with 'PHIS' from {1}".format(bnd_topo_fname, mpas_static_fname))
bnd_topo.variables['PHIS'][:] = mpas_static.variables['PHIS'][:]