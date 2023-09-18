#date: 2023-09-18T17:02:13Z
#url: https://api.github.com/gists/e0496d5766c12a0ae1738b943b41a536
#owner: https://api.github.com/users/matteoferla

# ##############################
# install conda at $APPTAINERENV_CONDA_PREFIX
# example usage:
: '
export DATA=/data/xchem-fragalysis
export APPTAINERENV_CONDA_PREFIX=$DATA/mferla/waconda
export JOB_SCRIPT=$DATA/shared/singularity.sh
export JOB_INNER_SCRIPT=/data/xchem-fragalysis/shared/install_conda.sh
condor_submit $DATA/shared/target_script.condor
'
# ##############################

# set -e
export DATA=/data/xchem-fragalysis

if [[ -z "$CONDA_PREFIX" ]]; then
    echo "Must provide CONDA_PREFIX in environment" 1>&2
    exit 1
fi

if ! [ -f $DATA/shared/Miniconda3-latest-Linux-x86_64.sh ]; then
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $DATA/shared/Miniconda3-latest-Linux-x86_64.sh;
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --output $DATA/shared/Miniconda3-latest-Linux-x86_64.sh;
fi;

# rm -r $CONDA_PREFIX
if [ -f "$CONDA_PREFIX" ]; then
bash $DATA/shared/Miniconda3-latest-Linux-x86_64.sh -p $CONDA_PREFIX -b -u
else
bash $DATA/shared/Miniconda3-latest-Linux-x86_64.sh -p $CONDA_PREFIX -b
fi

source $CONDA_PREFIX/etc/profile.d/conda.sh
export PIP_NO_CACHE_DIR=1
export PIP_NO_USER=1
export PYTHONUSERBASE=$CONDA_PREFIX
conda activate base
conda update -n base -y -c defaults conda
conda install -y -c conda-forge conda-libmamba-solver
conda config --set solver libmamba

# Jupyter stuff
conda install -y -n base -c conda-forge distro nodejs sqlite jupyterlab jupyter_http_over_ws nb_conda_kernels
conda update -y -c conda-forge nodejs   # peace among worlds
python -m pip install -q  jupyter_theme_editor

# install whatever you want here
python -m pip install -q pandas plotly seaborn pillow pandas pandarallel pandera nglview pebble rdkit jupyterlab-spellchecker;
conda install -y -n base -c conda-forge openssh nano;
conda install -y -n base -c conda-forge util-linux;
conda install -y -n base -c conda-forge openbabel plip git;
conda install -y -n base -c conda-forge -c bioconda kalign2 hhsuite muscle hhsuite mmseqs2;


python -m pip install -q fragmenstein  pyrosetta_help
python -m pip install -q $DATA/shared/pyrosetta-2023.27+release.e3ce6ea9faf-cp311-cp311-linux_x86_64.whl
# python -m pip cache purge  # PIP_NO_CACHE_DIR conflict


conda install -y  -c nvidia -c conda-forge cuda-toolkit cuda-nvcc cuda-command-line-tools gputil
conda install -y -c omnia -c conda-forge openmm openff-forcefields openff-toolkit  openmmforcefields
conda install -y -c pytorch -c conda-forge pytorch torchvision matplotlib pandas 


conda clean -y -t;
conda clean -y -i;

# A retro version for CentOS 7
CONDA_OVERRIDE_GLIBC=2.17 conda create -n glibc17 python=3.8;
# source $CONDA_PREFIX/etc/profile.d/conda.sh
# conda activate glibc17 # not base!


#chmod -R a+r $CONDA_PREFIX
#find $CONDA_PREFIX -type d -exec chmod 777 {} \;