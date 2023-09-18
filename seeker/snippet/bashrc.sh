#date: 2023-09-18T17:02:13Z
#url: https://api.github.com/gists/e0496d5766c12a0ae1738b943b41a536
#owner: https://api.github.com/users/matteoferla

# ========================
# Sets presents...
# tries to source $HOME/.bashrc
# or $HOME2/.bashrc;
# or a fallback
# ========================

# source /data/xchem-fragalysis/shared/bashrc.sh

# universal fixes
export PS1="[\u@\h \W]\$"
export LANG=en_GB.UTF-8
export DATA=/data/xchem-fragalysis
export HOST=${HOST:-$(hostname)}
export USER=${USER:-$(users)}
export HOME=${HOME:-$_CONDOR_SCRATCH_DIR}
export SHELL=/bin/bash
source /etc/os-release;
export PIP_NO_CACHE_DIR=1
export PIP_NO_USER=1
export NUMEXPR_MAX_THREADS=$(lscpu -p=CPU | tail -n 1 | xargs)
export GIPHY_API="KLlUojwaFWS2M7QWY8EzOl7LkIYPcCh2"
export MANIFOLD_API_KEY="v1:mAOYT8fFItGktDWNvo35vw"
export PLOTLY_API_KEY='y9O2jKKd5jsMBZM4aLz8'
export OE_LICENSE="$DATA/mferla/ASAP-oe_license.txt"
# frag network
export KUBECONFIG=$DATA/mferla/config-fragnet
export NEO4J_USER=matteo
export NEO4J_PASS='gone84shopping'
export USE_NEO4J_INSTEAD_API=true
# Jedi cache
mkdir -p $HOME2/.cache
export XDG_CACHE_HOME=$HOME2/.cache

# -------------------------------------

if [ -f $HOME/.bashrc ]
then
    source $HOME/.bashrc;
elif [ -f $HOME2/.bashrc ]
then
    source $HOME2/.bashrc;
else
    export HOME2=${HOME2:-/data/xchem-fragalysis/mferla}
        export PYTHONUSERBASE=${PYTHONUSERBASE:-$HOME2/conda/local}
    export CONDA_ENVS_PATH=${CONDA_ENVS_PATH:-$DATA/mferla/.conda/envs:$DATA/sanchezg/app/miniconda3_2/envs:$DATA/mferla/rocky-conda/envs}
    export MAMBA_ALWAYS_YES=yes
    
    source $DATA/mferla/rocky-conda/etc/profile.d/conda.sh
        conda activate

fi

export JUPYTER_CONFIG_DIR=${JUPYTER_CONFIG_DIR:-$HOME2/jupyter}
after_install() {
    conda clean --all -y 2>&1 > /dev/null; chmod -r -f a=rwX $CONDA_PREFIX 2>&1 > /dev/null;
}
sleep 1;
