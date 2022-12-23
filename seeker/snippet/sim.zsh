#date: 2022-12-23T16:55:58Z
#url: https://api.github.com/gists/eedefee6131d2b0f542af706086a1056
#owner: https://api.github.com/users/justin

#!/usr/bin/env zsh
#
# Convenience wrapper around the simctl command to perform operations related to iOS simulators.
# Author: Justin Williams (@justin)
#
# Usage: sim <options> <subcommand>
#
# This script is designed to work with ZSH. ymmv with other shells.

set -e

#######################################
# Print the help message and exit.
# Globals:
#   ZSH_ARGZERO
# Arguments:
#   None
# Outputs:
#   Writes help message to stdout.
#######################################
function printHelp() {
    echo "OVERVIEW: Perform operations related to iOS simulators.\n"

    echo "Usage: $ZSH_ARGZERO:t <options> <subcommand>\n"
    echo "Available commands:"
    echo
    echo "  booted                Prints the UUIDs of the currently booted simulators"
    echo "  create_device         Create a new simulator with the latest SDK (iphone or ipad)"
    echo "  delete_all_devices    Remove all simulator devices"
    echo "  open_container        Opens an app container for the currently booted simulators in the Finder"
    echo "  get_env               Prints the value of an environment variable from the booted simulator"
    echo "  open_url              Opens a URL in the booted simulator"
    echo "  add_media             Adds a media file to the booted simulator"
}

##############################################################
# Create a new simulator with currently selected SDK (via `xcode-select`).
# Globals:
#   None
# Arguments:
#   The simulator type to create. Defaults to "iphone".
#       iphone      Create an iPhone simulator (default).
#       ipad        Create an iPad simulator (default).
#   The optional name of the simulator to create. Defaults to the device type and runtime version.
# Outputs:
#   0 if the simulator was created, non-zero on error.
##############################################################
function create_device {
    local os=`xcrun simctl list runtimes | grep "com.apple.CoreSimulator.SimRuntime.iOS-16-" | tail -1 | awk '{print $2}'`
    local device_type="iPhone 14"
    local device_os="iOS ${os}"
    local device_name="${device_type} (${device_os})"
    local runtime=`xcrun simctl list runtimes | grep "com.apple.CoreSimulator.SimRuntime.iOS-16-" | tail -1 | awk '{print $7}'`

    if [[ $1 == "ipad" ]]; then
        device_type="iPad Air (5th generation)"
        device_os="iPadOS ${os}"
        device_name="${device_type} (${device_os})"
    fi

    if [[ $2 != "" ]]; then
        device_name="$2"
    fi

    echo "Creating ${device_name}..."
    xcrun simctl create "$device_name" "$device_type" "$runtime"
    exit 0
}

##############################################################
# Generate a list of booted simulators and return their UUIDs.
# Globals:
#   None
# Arguments:
#   None
# Outputs:
#   A list of UUIDs of the currently booted simulators.
##############################################################
function booted {
    xcrun simctl list devices | grep "Booted" | egrep -o -i "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    exit 0
}

##############################################################
# Remove all simulators from the system.
# Globals:
#   None
# Arguments:
#   None
# Outputs:
#   0 if all simulators were deleted, non-zero on error.
##############################################################
function removeAllSimulators {
    xcrun simctl list | egrep -o -i '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | xargs -L1 xcrun simctl delete
    exit 0
}

###########################################################################
# Get the path of the installed app's container and open it in the Finder.
# Globals:
#   None
# Arguments:
#   The app bundle identifier
#   The optional container type to open. Defaults to "app".
#       app                 The .app bundle
#       data                The application's data container
#       groups              The App Group containers
#       <group identifier>  A specific App Group container
# Outputs:
#   None
###########################################################################
function openContainer {
    open -R $(xcrun simctl get_app_container $(sim booted) $@)
}

###########################################################################
# Print an environment variable from the booted devices.
# Globals:
#   None
# Arguments:
#   The environment variable name
# Outputs:
#   None
###########################################################################
function getEnvironmentVariable {
    xcrun simctl getenv $(sim booted) $@
}

###########################################################################
# Open the passed in URL on the the booted devices.
# Globals:
#   None
# Arguments:
#   The URL to open in the booted simulators.
# Outputs:
#   None
###########################################################################
function openUrl {
    xcrun simctl openurl $(sim booted) $@
}

###########################################################################
# Copy the media file(s) from the passed in path to the booted devices.
# Globals:
#   None
# Arguments:
#   The path to a media file or folder of media files.
# Outputs:
#   None
###########################################################################
function addMedia {
    local files=(${(f)"$(find $1 -not -path '*/[@.]*' -type f)"})
    xcrun simctl addmedia $(sim booted) $files
}

case "$1" in
    booted)
        booted
        ;;
    create_device)
        create_device $2 $3
        ;;
    delete_all_devices)
        removeAllSimulators
        ;;
    open_container)
        openContainer $2 $3
        ;;
    get_env)
        getEnvironmentVariable $2
        ;;
    open_url)
        openUrl $2
        ;;
    add_media)
        addMedia $2
        ;;
    *)  # If we can't determine the proper argument, just print help.
        printHelp >&2
        exit 1
        ;;
esac
