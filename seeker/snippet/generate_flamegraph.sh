#date: 2025-01-08T16:56:49Z
#url: https://api.github.com/gists/8d520b84207800f1751a9376c366826a
#owner: https://api.github.com/users/noahebrooks-cb

#!/bin/bash

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit
fi

if ! command -v py-spy &> /dev/null; then
  echo "Installing py-spy..."
  pip install py-spy
fi

if [ ! -f ./flamegraph.pl ]; then
  curl -O https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl
  chmod +x flamegraph.pl
fi

read -p "Enter the PID of the Python process to profile: " PID

OUTPUT_FILE="flamegraph_$(date +%Y%m%d%H%M%S).svg"
echo "Generating flame graph for PID $PID over 15 seconds..."
py-spy record -p $PID --output $OUTPUT_FILE --duration 15

if [ -f "$OUTPUT_FILE" ]; then
  echo "Flame graph generated: $OUTPUT_FILE"
else
  echo "Failed to generate flame graph."
fi