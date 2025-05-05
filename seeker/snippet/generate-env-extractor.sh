#date: 2025-05-05T16:43:09Z
#url: https://api.github.com/gists/4f170e1b5233b10db3d69510a9b877c5
#owner: https://api.github.com/users/ersanyamarya

#!/bin/bash

# This script generates a COMMAND that can be used to extract the values of environment variables from a template file.
# The template file should contain variable names in the format NAME=value, NAME="value", NAME='value', or NAME=${value}.
# This script will create a command that can be copied and pasted into a shell to generate the env file from the environment based on the template.

# Function to display usage information
show_usage() {
    echo "Usage: $(basename "$0") [-h] [-o output_file] template_file"
    echo
    echo "Options:"
    echo "  -h           Show this help message"
    echo "  -o file     Write output directly to the specified file"
    echo "  template_file  Path to the template env file"
    echo
    echo "Example:"
    echo "  $(basename "$0") .env.template"
    echo "  $(basename "$0") -o .env .env.template"
}

# Process command line arguments
output_file=""
while getopts "ho:" opt; do
    case $opt in
    h)
        show_usage
        exit 0
        ;;
    o)
        output_file="$OPTARG"
        ;;
    \?)
        echo "Invalid option: -$OPTARG" >&2
        show_usage
        exit 1
        ;;
    esac
done

# Shift the options so $1 becomes the template file
shift $((OPTIND - 1))

if [ -z "$1" ]; then
    echo "Error: No template file specified." >&2
    show_usage
    exit 1
fi

# Check if running in bash
if [ -z "$BASH_VERSION" ]; then
    echo "Error: This script must be run with bash. Please run it with bash." >&2
    exit 1
fi

# Check if the template file exists
if [ ! -f "$1" ]; then
    echo "Error: Template file '$1' not found." >&2
    exit 1
fi

# Check if the template file is readable
if [ ! -r "$1" ]; then
    echo "Error: Template file '$1' is not readable." >&2
    exit 1
fi

# Extract variable names from the template file
# This handles variables in formats: NAME=value, NAME="value", NAME='value', NAME=${value}
vars=$(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' "$1" | sed 's/=$//' | tr '\n' ' ')

if [ -z "$vars" ]; then
    echo "Warning: No environment variables found in template file." >&2
    exit 0
fi

# Generate the command
base_command="for var in \"\$@\"; do printenv \"\$var\" >/dev/null && echo \"\$var=\$(printenv \"\$var\")\" || echo \"# \$var is not set\" >&2; done"
if [ -n "$output_file" ]; then
    # If output file is specified, generate command that writes directly to the file
    command="bash -c '(echo \"#!/usr/bin/env bash\"; echo; $base_command) > \"$output_file\"' -- $vars"
    echo
    echo "Writing environment variables to '$output_file'..."
    eval "$command"
    echo "Done! Generated environment file at '$output_file'"
else
    # Generate command for manual copying
    command="bash -c '$base_command | awk '\\''NR==1{print \"#!/usr/bin/env bash\\n\"} {print}'\\''' -- $vars"
    echo
    echo "Command to extract environment variables from '$1':"
    echo
    echo "$command"
    echo
    echo "To use this command:"
    echo "1. Copy the above command."
    echo "2. Paste and run it in your shell to generate the content of the environment file."
fi
