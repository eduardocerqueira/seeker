#date: 2023-09-18T17:02:13Z
#url: https://api.github.com/gists/e0496d5766c12a0ae1738b943b41a536
#owner: https://api.github.com/users/matteoferla

#!/bin/bash

# ========================
# JUMP!
# Run a permanent ssh reverse proxy connection on SSH_FORWARD_PORT
# Requires SSH_USER remote user name
# A folder SSH_TMP_FOLDER which will be moved to $HOME/.ssh/
# which includes the filename SSH_KEY
# ========================

# $SSH_USER
if [ -z "$SSH_USER" ]; then
    crash brutally "Your remote username SSH_USER ($SSH_USER) is not specified"
fi
if [ -z "$SSH_GATE_ADDRESS" ]; then
    crash brutally "Your remote username SSH_GATE_ADDRESS ($SSH_GATE_ADDRESS) is not specified"
fi
if [ -z "$SSH_INNER_ADDRESS" ]; then
    crash brutally "Your remote username SSH_INNER_ADDRESS ($SSH_INNER_ADDRESS) is not specified"
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
#export SSH_PORT=${SSH_PORT:-22}

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
ssh-keygen -R $SSH_GATE_ADDRESS -f "$SSH_FOLDER/known_hosts"
while true;
do
ssh -N -R 0.0.0.0:$SSH_FORWARD_PORT:0.0.0.0:$SSH_FORWARD_PORT \
-o ProxyCommand="ssh -v -W %h:%p -l $SSH_USER -i $SSH_FOLDER/$SSH_KEY \
-o StrictHostKeyChecking=no \
$SSH_GATE_ADDRESS" \
-i $SSH_FOLDER/$SSH_KEY \
-o ServerAliveInterval=180 \
-o UserKnownHostsFile=$SSH_FOLDER/known_hosts \
-l $SSH_USER \
-o ExitOnForwardFailure=yes \
-o StrictHostKeyChecking=no \
$SSH_INNER_ADDRESS \
-v;

echo 'Connection to $SSH_GATE_ADDRESS > $SSH_INNER_ADDRESS lost' 1>&2;
sleep 600;
done;

