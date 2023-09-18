#date: 2023-09-18T17:02:13Z
#url: https://api.github.com/gists/e0496d5766c12a0ae1738b943b41a536
#owner: https://api.github.com/users/matteoferla

# ##############################
# example usage:
: '
export DATA=/data/xchem-fragalysis
export APPTAINERENV_CONDA_PREFIX=$DATA/mferla/waconda
export JOB_SCRIPT=$DATA/shared/singularity.sh
export JOB_INIT_SCRIPT=/data/xchem-fragalysis/shared/stats_connection.sh
export JOB_INNER_SCRIPT=/data/xchem-fragalysis/shared/notebook.sh
export JOB_PORT=1300
export SSH_FORWARD_PORT=1300
export SSH_KEY=seiryu
export SSH_USER=ferla
export SSH_FOLDER=$DATA/mferla/singularity/tmp
export JUPYTER_CONFIG_DIR=$DATA/mferla/jupyter 
export APPTAINERENV_CONDA_PREFIX=/data/xchem-fragalysis/mferla/waconda
export APPTAINERENV_JUPYTER_notebook_dir=/data/xchem-fragalysis/mferla
export APPTAINERENV_JUPYTER_CONFIG_DIR=$JUPYTER_CONFIG_DIR
export APPTAINER_HOSTNAME='lucky13'
condor_submit $DATA/shared/target_script.condor -a 'Requirements=(machine == "orpheus-worker-gpu-13.novalocal")'
'
# ##############################



export HOST=${HOST:-$(hostname)}
export USER=${USER:-$(users)}
export HOME=${HOME:-$_CONDOR_SCRATCH_DIR}
source /etc/os-release;

if [ -n "$JUPYTER_PORT" ]; then
    echo "$JUPYTER_PORT set"
elif [ -n "$JOB_PORT" ]; then
    export JUPYTER_PORT=$JOB_PORT
elif [ -n "$SSH_FORWARD_PORT" ]; then
    export JUPYTER_PORT=$SSH_FORWARD_PORT
else
    raise error "Your JUPYTER_PORT is not specified"
fi

if [ -z "$JUPYTER_CONFIG_DIR" ]; then
    raise error "Your JUPYTER_CONFIG_DIR is not specified either"
fi



echo "************************"
echo "HELLO JUPYTER!"
echo "************************"
echo "Greet from Jupyter lab script ${0} as $USER in $HOST which runs $PRETTY_NAME on $JUPYTER_PORT with settings from $JUPYTER_CONFIG_DIR"

source /data/xchem-fragalysis/shared/bashrc.sh;
# conda activate

#export JUPYTER_CONFIG_PATH=$HEADQUARTERS/.jupyter

# First time? Remember to set:
# jupyter notebook --generate-config
# yes invasion | jupyter server password

# port is JUPYTER_PORT
while true
do
jupyter lab --ip="0.0.0.0" --no-browser
done
