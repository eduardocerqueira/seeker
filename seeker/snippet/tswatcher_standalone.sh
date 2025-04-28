#date: 2025-04-28T17:00:28Z
#url: https://api.github.com/gists/60c16b7b0d321619b8b06c7cdad822ee
#owner: https://api.github.com/users/yigtuyumz

#!/usr/bin/env bash

# Function to print usage
print_usage() {
  echo 'Usage: Provide a filename with a .ts extension to compile and watch.'
  exit 1
}

# Function to check required binaries
check_binaries() {
  local required_binaries
  required_binaries="nodemon tsc"

  for binary in ${required_binaries}; do
    echo -ne "Checking ${binary}... "
    if ! which "${binary}" > /dev/null 2>&1; then
      echo "NOK!"
      echo "Please install '${binary}' as a superuser and then try again."
      echo "  >  npm install -g ${binary}"
      exit 1
    else
      echo "OK!"
    fi
  done
}

# Function to check if file exists
check_file_exists() {
  local file_name
  file_name="${1}"
  if  [ ! -f "${file_name}" ] ||
        [ -d "${file_name}" ] ||
        [ -L "${file_name}" ]; then
    echo "File '${file_name}' does not exist, exiting."
    exit 1
  fi
}

# Function to start nodemon
start_nodemon() {
  local file_name
  file_name="${1}"
  echo 'Starting nodemon...'
  nodemon -q "${file_name}" &
}

# Function to start TypeScript compilation with --watch
start_tsc() {
  local file_name
  file_name="${1}"
  echo 'Starting TypeScript compiler (tsc)...'
  tsc -t esnext --watch "${file_name}" &
}

# Function to handle cleanup when script exists
cleanup() {
  printf "\r%s\n" "Killing background jobs..."
  local child_pids
  child_pids="$(jobs -p)"
  for pid in ${child_pids}; do
    echo "Killing process with PID: ${pid}"
    kill "${pid}"
  done
  stty sane
}

main() {
  # Check if a filename argument is provided
  if [ -z "${1}" ]; then
    print_usage
  fi

  local file_name
  file_name="${1}"

  # Check the required binaries which are nodemon and tsc
  check_binaries
  # Check if provided file exists
  check_file_exists "${file_name}"

  # Set up a trap to clean up on exit
  trap cleanup SIGINT SIGTERM

  # Start the processes
  start_tsc "${file_name}"
  start_nodemon "${file_name}"

  # Wait for all background jobs to finish
  wait
}

# Run the main function
main "${1}"
