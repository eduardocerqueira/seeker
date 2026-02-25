#date: 2026-02-25T17:45:29Z
#url: https://api.github.com/gists/eb2477b0ca3f0694fcb92d49d19064e7
#owner: https://api.github.com/users/miloridenour

#!/bin/bash

# Base script content
# Exit immediately on error, unset variables are errors, and fail on any command in a pipeline
set -euo pipefail

# Default values
DOCKER_ARGS=()
CLIVE_ARGS=()
CLIVE_DATA_DIR="${HOME}/.clive"
IMAGE_NAME="hiveio/clive:v1.28.1.1"
TTY_ARGS="-it"
PIPELINE=""

if ! [ -t 0 ]; then
    read -r PIPELINE
    TTY_ARGS="-i"
fi

# Add Docker argument to the Docker args array
add_docker_arg() {
    local arg="$1"
    DOCKER_ARGS+=("$arg")
}

# Add Clive argument to the Clive args array
add_clive_arg() {
    local arg="$1"
    CLIVE_ARGS+=("$arg")
}

# Check if Docker is installed
check_docker_installed() {
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed."
        echo "Please consult Docker installation documentation: https://docs.docker.com/engine/install/ubuntu/"
        exit 1
    fi
}

# Parse command-line arguments
parse_args() {
  common_parse_args "$@"
}

common_parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --docker-option=*)
                local options_string="${1#*=}"
                IFS=" " read -ra options <<< "$options_string"
                for option in "${options[@]}"; do
                    add_docker_arg "$option"
                done
                ;;
            --docker-option)
                shift
                local options_string="$1"
                IFS=" " read -ra options <<< "$options_string"
                for option in "${options[@]}"; do
                    add_docker_arg "$option"
                done
                ;;
            --data-dir=*)
                CLIVE_DATA_DIR="${1#*=}"
                ;;
            --data-dir)
                shift
                CLIVE_DATA_DIR="$1"
                ;;
            --help)
                print_help
                exit 0
                ;;
            -*)
                add_clive_arg "$1"
                ;;
            *)
                IMAGE_NAME="$1"
                echo "Using image name: $IMAGE_NAME"
                ;;
        esac
        shift
    done

    # Collect remaining command-line arguments and Clive-specific arguments
    CMD_ARGS=("$@")
    CMD_ARGS+=("${CLIVE_ARGS[@]}")
}

common_validation(){
    validate_data_dir
}

# Validate and create the Clive data directory
validate_data_dir() {
    if [ -z "${CLIVE_DATA_DIR}" ]; then
        echo "Error: Missing --data-dir argument"
        exit 2
    fi

    mkdir -p "${CLIVE_DATA_DIR}"
    chmod -R 700 "${CLIVE_DATA_DIR}"

    # Get absolute path for the data directory
    CLIVE_DATA_ABS_DIR=$(realpath "${CLIVE_DATA_DIR}")
}

# Configure Docker volume mappings based on the Clive data directory
configure_docker_volumes() {
    add_docker_arg "--volume"
    add_docker_arg "${CLIVE_DATA_ABS_DIR}:/clive/.clive/"

    CLIVE_DATA_LOCATION_DIR=$(realpath "${CLIVE_DATA_DIR}/../")
    HOME_ABS_DIR=$(realpath "${HOME}")

    # If Clive directory is inside the user's home, add an additional volume mapping
    if [[ "${CLIVE_DATA_LOCATION_DIR}/" = "${HOME_ABS_DIR}"/* ]]; then
        add_docker_arg "--volume"
        add_docker_arg "${CLIVE_DATA_LOCATION_DIR}:${CLIVE_DATA_LOCATION_DIR}"
    fi
}

# Function to clean up (stop container) on exit signals
cleanup() {
    echo "Stopping container ${CONTAINER_NAME}...."
    docker stop "$CONTAINER_NAME"
    echo "Cleanup actions done."
}
trap cleanup HUP INT QUIT TERM

# Run the Docker container
run_docker_container() {
    DOCKER_RUN_ARGS=(
        --name="${CONTAINER_NAME}"
        --detach-keys='ctrl-@,ctrl-q'
        --rm ${TTY_ARGS}
        -e CLIVE_UID="$(id -u)"
        --stop-timeout=180
        "${DOCKER_ARGS[@]}"
        "${IMAGE_NAME}"
        "${CMD_ARGS[@]}"
    )

    if [[ -n "${PIPELINE:-}" ]]; then
        echo "${PIPELINE}" | docker run "${DOCKER_RUN_ARGS[@]}"
    else
        docker run "${DOCKER_RUN_ARGS[@]}"
    fi
}

# Main script execution
main() {
    check_docker_installed
    parse_args "$@"
    validate
    configure_docker_volumes
    run_docker_container
}

# Start clive cli script content

# Set the container name for CLI mode
CONTAINER_NAME="clive-cli-$(date +%s)"
export CONTAINER_NAME

# Set HAS_EXEC variable when passing --exec flag, this variable is used for validation purpose
HAS_EXEC=0
# Set HAS_PROFILE_NAME variable when passing --profile-name flag, this variable is used for validation purpose
HAS_PROFILE_NAME=0

# Print usage information for the script
print_help() {
    echo "Usage: $0 [<docker_img>] [OPTION[=VALUE]]... [<clive_option>]..."
    echo
    echo "Allows to start Clive application in CLI mode."
    echo "CLI terminal will start in mapped host directory."
    echo
    echo "OPTIONS:"
    echo "  --data-dir=DIRECTORY_PATH      Points to a Clive data directory to store profile data. Defaults to ${HOME}/.clive directory."
    echo "  --docker-option=OPTION         Allows specifying additional Docker options to pass to underlying Docker run."
    echo "  --exec=PATH_TO_FILE            Path to bash script to be executed."
    echo "  --profile-name=PROFILE_NAME    Name of profile that will be used, default is profile selection."
    echo "  --unlock-time=MINUTES          Unlock time in minutes, default is no timeout for unlock."
    echo "  --help                         Display this help screen and exit."
    echo
}

# Get the WORKINGDIR from containers image
get_workdir() {
    docker inspect --format='{{.Config.WorkingDir}}' "$IMAGE_NAME"
}

# Helper function for exec argument
handle_exec_option() {
    local file_to_mount="$1"

    WORKDIR=$(get_workdir)
    if [ -z "$WORKDIR" ]; then
        echo "Error: Could not retrieve WORKDIR for image: $IMAGE_NAME"
        exit 1
    fi

    FILE_TO_MOUNT_NAME=$(basename "$file_to_mount")

    MOUNT_TARGET="${WORKDIR}/${FILE_TO_MOUNT_NAME}"
    add_docker_arg "--volume"
    add_docker_arg "${file_to_mount}:${MOUNT_TARGET}"
    add_clive_arg "--exec"
    add_clive_arg "${FILE_TO_MOUNT_NAME}"
}

# Helper function for --profile-name argument
handle_profile_name_option() {
    local profile_name="$1"
    add_clive_arg "--profile-name"
    add_clive_arg "${profile_name}"
}

# Helper function for --unlock-time argument
handle_unlock_time_option() {
    local unlock_time_mins="$1"
    add_clive_arg "--unlock-time"
    add_clive_arg "${unlock_time_mins}"
}

# Override parse_args to handle --exec flag specifically for CLI mode
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --exec)
                shift
                handle_exec_option "${1}"
                HAS_EXEC=1
                ;;
            --exec=*)
                handle_exec_option "${1#*=}"
                HAS_EXEC=1
                ;;
            --profile-name)
                shift
                handle_profile_name_option "${1}"
                HAS_PROFILE_NAME=1
                ;;
            --profile-name=*)
                handle_profile_name_option "${1#*=}"
                HAS_PROFILE_NAME=1
                ;;
            --unlock-time)
                shift
                handle_unlock_time_option "${1}"
                ;;
            --unlock-time=*)
                handle_unlock_time_option "${1#*=}"
                ;;
            *)
            new_args+=("$1")
            ;;
        esac
        shift
    done
    common_parse_args "${new_args[@]}"
}

validate_pipeline_with_exec() {
    # pipeline input is only allowed with --exec
    if [[ -n "${PIPELINE:-}" && "$HAS_EXEC" -eq 0 ]]; then
        echo "Error: Pipeline input is only allowed when --exec is passed."
        exit 3
    fi
}

validate_profile_name_with_exec() {
    # combining flag --profile-name with --exec is forbidden
    if [[ "$HAS_PROFILE_NAME" -eq 1 && "$HAS_EXEC" -eq 1 ]]; then
        echo "Error: Cannot set both --profile-name and --exec."
        exit 3
    fi
}

validate() {
    validate_pipeline_with_exec
    validate_profile_name_with_exec
    common_validation
}

# Add the --cli flag to Clive arguments
add_clive_arg "--cli"

# Run the main function, passing the extended parse_args
main "$@"
