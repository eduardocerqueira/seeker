#date: 2024-11-06T17:11:53Z
#url: https://api.github.com/gists/c8c67f9e1be4834d382ed349a8b5c259
#owner: https://api.github.com/users/talalashraf

# Add this function to bashrc or zshrc on macos. Run before unplugging a dock.

leave() {
    local failed_mounts=()
    declare -i success_count=0
    declare -i total_count=0

    echo "Preparing to safely unmount all external volumes..."

    # Get list of mounted volumes that aren't the system volume
    for volume in /Volumes/*; do
        # Skip if not a directory or if it's the system volume
        # Note: Checks for both "Macintosh HD" and localized system volume names
        if [[ ! -d "$volume" || "$volume" == "/Volumes/Macintosh HD"* || "$volume" == "/Volumes/Data" ]]; then
            continue
        fi

        ((total_count++))
        echo "Unmounting: $volume"
        diskutil eject "$volume" > /dev/null 2>&1

        if [ $? -eq 0 ]; then
            ((success_count++))
        else
            failed_mounts+=("$volume")
        fi
    done

    # Report results
    if [ ${#failed_mounts[@]} -eq 0 ]; then
        echo "All external volumes safely unmounted. You can now unplug your devices."
        return 0
    else
        echo "Warning: Failed to unmount the following volumes:"
        printf '%s\n' "${failed_mounts[@]}"
        echo "Please close any programs using these volumes and try again."
        return 1
    fi
}