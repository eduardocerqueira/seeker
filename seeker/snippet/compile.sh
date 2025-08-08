#date: 2025-08-08T17:10:31Z
#url: https://api.github.com/gists/4277f7036e915d2e5610338b1655a413
#owner: https://api.github.com/users/uturuncoglu

# This is using UFS spack stack to access ESMF and netCDF installations
module purge
module use /glade/work/turuncu/COASTAL/s111/ufs-weather-model/modulefiles
module load ufs_derecho.intel
module load cmake
module li

rm -f unstmesh.x *.o
ifort -g -traceback -o unstmesh.x unstmesh.f90 -L$netcdf_fortran_ROOT/lib -lnetcdff -I$netcdf_fortran_ROOT/include