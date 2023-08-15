#date: 2023-08-15T16:54:34Z
#url: https://api.github.com/gists/b3709239b483b172834f697dc1f4f852
#owner: https://api.github.com/users/trvswgnr

#!/bin/sh

main() {
    check_exists "$local_dir" "local directory"
    check_exists "$ssh_key" "ssh key file"
    log_message "starting backup of $local_dir to $remote_server:$remote_dir"
    rsync -avz \
        -e "ssh -i $ssh_key" --delete --log-file="$log_file" \
        --progress "$local_dir" "$remote_server:$remote_dir" \
        || handle_error "backup failed"
    log_message "backup completed successfully"
}

local_dir="/path/to/local"
remote_dir="/path/to/remote"
remote_server="user@server"
log_file="/path/to/log"
ssh_key="/path/to/private"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$log_file"
}

handle_error() {
    log_message "error: $1"
    exit 1
}

check_exists() {
    if [ ! -e "$1" ]; then
        handle_error "$2 $1 does not exist."
    fi
}

main