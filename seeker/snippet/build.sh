#date: 2024-12-02T16:47:43Z
#url: https://api.github.com/gists/56cf05f5d9c781eda599ae1e80f7b691
#owner: https://api.github.com/users/BaoZhuhan

#!/bin/bash
export ROOT_DIR=$(pwd)
export MODULE_DIR=/etc/profile.d
# export PATH=~/env/cmake/bin:$PATH
# export oneAPI_DIR= !the correct path for oneAPI!

# source $MODULE_DIR/module.sh  
# source $oneAPI_DIR/setvars.sh
# module load cmake 

mkdir Build
ROOT_DIR/Build
# Unzip all the files
tar -zxvf $ROOT_DIR/lapack-3.11.tar.gz -C $ROOT_DIR/Build/
tar -zxvf $ROOT_DIR/parmetis-4.0.3.tar.gz -C $ROOT_DIR/Build/
tar -zxvf $ROOT_DIR/hypre-2.28.0.tar.gz -C $ROOT_DIR/Build/
tar -zxvf $ROOT_DIR/petsc-3.19.3.tar.gz -C $ROOT_DIR/Build/
tar -zxvf $ROOT_DIR/petsc_solver.tar.gz -C $ROOT_DIR/Build/
tar -zxvf $ROOT_DIR/OpenCAEPoro.tar.gz -C $ROOT_DIR/Build/

# Build LAPACK

cd $ROOT_DIR/Build/lapack-3.11

make blaslib
make cblaslib
make lapacklib
make lapackelib

# Build ParMETIS
cd $ROOT_DIR/Build/parmetis-4.0.3

make config cc=mpiicx cxx=mpiicpx prefix=$ROOT_DIR/Build/parmetis-4.0.3/parmetis-install
make -j 16
make install

# Build Hypre

cd $ROOT_DIR/Build/hypre-2.28.0/

cd src/
make clean
./configure cc=mpiicx cxx=mpiicpx --prefix="$ROOT_DIR/Build/hypre-2.28.0/install" --with-MPI --enable-shared
make -j 16
make install

# Build PETSc

cd $ROOT_DIR/Build/petsc-3.19.3

export PETSC_DIR=$ROOT_DIR/Build/petsc-3.19.3
export PETSC_ARCH=petsc_install

./configure cc=mpiicx cxx=mpiicpx \
	--with-fortran-bindings=0 \
	--with-hypre-dir=$ROOT_DIR/Build/hypre-2.28.0/install \
	--with-debugging=0 \
	COPTFLAGS="-O3" \
	CXXOPTFLAGS="-O3" \

make -j 20 PETSC_DIR=$ROOT_DIR/Build/petsc-3.19.3 PETSC_ARCH=petsc_install all
make all check

# Build PETSc Solver

cd $ROOT_DIR/Build/petsc_solver

export CC=mpiicx
export CXX=mpiicpx

export CPATH=$ROOT_DIR/Build/lapack-3.11/CBLAS/include:$ROOT_DIR/Build/lapack-3.11/LAPACKE/include:$CPATH
export LD_LIBRARY_PATH=$ROOT_DIR/Build/lapack-3.11:$LD_LIBRARY_PATH

rm -rf build 
mkdir build
cd build
cmake ..
make
mv libpetsc_solver.a ../lib/

# Build OpenCAEPoro

cd $ROOT_DIR/Build/OpenCAEPoro

export PARMETIS_DIR=$ROOT_DIR/Build/parmetis-4.0.3
export PARMETIS_BUILD_DIR=$ROOT_DIR/Build/parmetis-4.0.3/build/Linux-x86_64
export METIS_DIR=$ROOT_DIR/Build/parmetis-4.0.3/metis
export METIS_BUILD_DIR=$ROOT_DIR/Build/parmetis-4.0.3/build/Linux-x86_64
export PETSC_DIR=$ROOT_DIR/Build/petsc-3.19.3
export PETSC_ARCH=petsc_install
export PETSCSOLVER_DIR=$ROOT_DIR/Build/petsc_solver
export CPATH=$ROOT_DIR/Build/petsc-3.19.3/include/:$CPATH
export CPATH=$ROOT_DIR/Build/petsc-3.19.3/petsc_install/include/:$ROOT_DIR/Build/parmetis-4.0.3/metis/include:$ROOT_DIR/Build/parmetis-4.0.3/include:$CPATH
export CPATH=$ROOT_DIR/Build/lapack-3.11/CBLAS/include/:$CPATH


rm -fr build; mkdir build; cd build;

echo "cmake -DUSE_PETSCSOLVER=ON -DUSE_PARMETIS=ON -DUSE_METIS=ON -DCMAKE_VERBOSE_MAKEFILE=OFF -DCMAKE_BUILD_TYPE=Release .."
cmake -DUSE_PETSCSOLVER=ON -DUSE_PARMETIS=ON -DUSE_METIS=ON -DCMAKE_VERBOSE_MAKEFILE=OFF -DCMAKE_BUILD_TYPE=Release ..

echo "make -j 32"
make -j 32

echo "make install"
make install

# Run the test
cd $ROOT_DIR/Build/

mpirun -n 4 ./testOpenCAEPoro ./data/test/test.data