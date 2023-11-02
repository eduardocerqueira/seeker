#date: 2023-11-02T16:49:51Z
#url: https://api.github.com/gists/3fadf04298ddb8a9c255e0c10c6d997c
#owner: https://api.github.com/users/albertbuchard

tailast() {
    # Get the directory from the argument or use the current directory as default
    local dir="${1:-.}"

    # Find the most recently modified file in the directory without descending into subdirectories
    local latest_file=$(find "$dir" -maxdepth 1 -type f -exec stat --format='%Y %n' {} \; | sort -n | tail -1 | awk '{print $2}')

    # Check if a file was found
    if [[ -z "$latest_file" ]]; then
        echo "No files found in $dir"
        return 1
    fi

    # Echo and tail the file
    echo "Tailing the file: $latest_file"
    tail -f "$latest_file"
}


watchjob() {
    local job_id="$1"

    # Check if job_id is provided
    if [[ -z "$job_id" ]]; then
        echo "Please provide a job ID."
        return 1
    fi

    # Use watch to repeatedly execute scontrol show job
    watch -n 1 "scontrol show job $job_id"
}

git_sbatch() {
    # Name of the script file you want to submit with sbatch
    SCRIPT_FILE=$1
    shift  # Shift arguments to remove the script filename

    # Capture the current git commit hash
    COMMIT_HASH=$(git rev-parse HEAD)

    # Create a temporary script with git checkout and original commands in the tmp directory
    TMP_SCRIPT="./tmp_sbatch_script_$(date +%s).sh"

    # Add the shebang
    echo "#!/bin/bash" > $TMP_SCRIPT

    # Extract the SBATCH directives from the script file
    grep "#SBATCH" $SCRIPT_FILE >> $TMP_SCRIPT

    # Add the git checkout command
    echo "git checkout $COMMIT_HASH" >> $TMP_SCRIPT

    # Append the rest of the script after removing the SBATCH directives and the shebang
    grep -v "#SBATCH" $SCRIPT_FILE | grep -v "#!/bin/bash" >> $TMP_SCRIPT

    # Submit the temporary script to sbatch with additional parameters
    sbatch $TMP_SCRIPT "$@"

    # Remove the temporary script after submission
    rm $TMP_SCRIPT
}