#date: 2025-08-08T16:48:50Z
#url: https://api.github.com/gists/fe33d7069a28ced435485a1250fee23e
#owner: https://api.github.com/users/digitalsparky

#!/bin/bash
set -eo pipefail

### CONFIGURATION STARTS HERE

# Command and arguments to run
DAEMON="sleep 3600"

# Name of the cgroup to use
DAEMONCGROUP="myapp"

# Standard logging destination (/dev/stdout)
LOGFILE="/dev/stdout"

# Error logging destination (/dev/stderr)
ERRLOGFILE="/dev/stderr"

# Graceful shutdown channel path (named pipe)
GRACEFUL_SHUTDOWN_CHANNEL=".$0.ch" # this is a fifo, it will be created in the $PWD by default and will be cleaned up after.

### CONFIGURATION ENDS HERE

# SET DEFAULT DAEMON PID
DAEMONPID=0

### FUNCTIONS STARTS HERE

# Create a named pipe if it doesn't exist
function create_named_pipe(){
    local pipe_name=$1
    if [[ ! -p "$pipe_name" ]]; then
        mkfifo $pipe_name
    fi
}

# Handle SIGKILL signal
function _kill(){
    echo "Caught SIGKILL signal!"
    kill -s SIGKILL $DAEMONPID
    _cleanup
}

# Handle SIGTERM and SIGINT signals
function shutdown(){
    echo "Caught shutdown signal!"
    kill -s SIGTERM $DAEMONPID
    _cleanup
    echo "Process was gracefully shutdown"
}

# Cleanup named pipe, etc
function _cleanup() {
    unlink $GRACEFUL_SHUTDOWN_CHANNEL
}

# Main function
function main() {

    # Create named pipe if it doesn't exist
    create_named_pipe $GRACEFUL_SHUTDOWN_CHANNEL

    # Run daemon, redirecting stdout and stderr to log files, then send process to the background
    $DAEMON >$LOGFILE 2 > $ERRLOGFILE &

    # Store daemon PID
    DAEMONPID=$!

    echo "DAEMON PID IS $DAEMONPID"

    # Set the $DAEMONCGROUP cgroup to the daemon PID
    echo $DAEMONPID > /sys/fs/cgroup/$DAEMONCGROUP/cgroup.procs

    # Wait for the daemon to finish
    < $GRACEFUL_SHUTDOWN_CHANNEL
    echo "Process has shut down"
}

### FUNCTIONS ENDS HERE

# Handle signals
trap shutdown SIGINT SIGTERM
trap _kill SIGKILL

# Run main function
main
