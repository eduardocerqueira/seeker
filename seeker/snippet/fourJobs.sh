#date: 2023-07-26T16:55:02Z
#url: https://api.github.com/gists/662aa6034cc6c90ba387fe46b5018c53
#owner: https://api.github.com/users/jimdiroffii

#!/bin/bash

# Manages concurrent execution of nohup jobs with a maximum limit.

# Number of maximum concurrent jobs
MAX_JOBS=4

# List of commands you want to run with nohup
declare -a commands=(
    "./sleepTest.sh"
    "./sleepTest.sh"
    "./sleepTest.sh"
    "./sleepTest.sh"
    "./sleepTest.sh"
    "./sleepTest.sh"
    "./sleepTest.sh"
    # ... add more commands as needed
)

# Function to get the current number of background jobs
num_jobs() {
    jobs -p | wc -l
}

# Loop through each command and execute them
for cmd in "${commands[@]}"; do
    while true; do
        # Check if the number of current jobs is less than the maximum allowed
        if [[ $(num_jobs) -lt $MAX_JOBS ]]; then
            echo "Executing: nohup $cmd & $(($(num_jobs) + 1)) now running"
            nohup $cmd &> /dev/null &
            sleep 1  # give a little time before checking again
            break
        fi

        # Wait a bit before rechecking
        sleep 5
    done
done

# Wait for all jobs to finish
wait