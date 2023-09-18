#date: 2023-09-18T17:02:13Z
#url: https://api.github.com/gists/e0496d5766c12a0ae1738b943b41a536
#owner: https://api.github.com/users/matteoferla

if [ -z "$SSH_USER" ]; then
    raise error "Your remote username SSH_USER ($SSH_USER) is not specified"
fi
if [ -z "$SSH_ADDRESS" ]; then
    raise error "Your remote address SSH_ADDRESS ($SSH_ADDRESS) is not specified"
fi
if [ -n "$SSH_FORWARD_PORT" ]; then
    echo '$SSH_FORWARD_PORT provided directly.'
elif [ -n "$JOB_PORT" ]; then
    export SSH_FORWARD_PORT=$JOB_PORT
elif [ -n "$JUPYTER_PORT" ]; then
    export $SSH_FORWARD_PORT=$JUPYTER_PORT
elif [ -n "$APPTAINERENV_JUPYTER_PORT" ]; then
    export $SSH_FORWARD_PORT=$APPTAINERENV_SSH_FORWARD_PORT
else
    raise error 'No $SSH_FORWARD_PORT provided'
fi

export DATA=/data/xchem-fragalysis;
export SSH_KEY=${SSH_KEY:-*}
export SSH_FOLDER=${SSH_FOLDER:-$HOME/.ssh}
export SSH_PORT=${SSH_PORT:-22}

# most applications are okay with path//path but not ssh
export SSH_FOLDER=$(echo "$SSH_FOLDER" | sed "s/\/\//\//g" | sed "s/\/$//")

touch $SSH_FOLDER/test.txt
if [ ! -f $SSH_FOLDER/test.txt ]
then
	echo "The folder $SSH_FOLDER is inaccessible"
        mkdir -p /tmp/ssh
        export SSH_FOLDER=/tmp/ssh
fi

echo 'prep connections by moving keys from $SSH_FOLDER to $HOME'
mkdir -p $SSH_FOLDER
touch $SSH_FOLDER/known_hosts
chmod 700 $SSH_FOLDER
chmod 600 $SSH_FOLDER/*

echo 'accepting fingerprints'
ssh-keygen -R $SSH_ADDRESS -f "$SSH_FOLDER/known_hosts"
while true;
do
ssh -N -R 0.0.0.0:$SSH_FORWARD_PORT:localhost:$SSH_FORWARD_PORT -p 666 \
    -o ServerAliveInterval=180 \
    -o ExitOnForwardFailure=yes \
    -i $SSH_FOLDER/$SSH_KEY \
    -o UserKnownHostsFile=$SSH_FOLDER/known_hosts \
    -l $SSH_USER \
    -p $SSH_PORT \
$SSH_ADDRESS;

echo 'Connection to stats lost' 1>&2;
sleep 600;
done;


mkdir -p $HOME/.ssh/
mv singularity/tmp/* $HOME/.ssh/
#ssh-keygen -R www.matteoferla.com:666 -f $HOME/.ssh/
while true
do

sleep 60;
done