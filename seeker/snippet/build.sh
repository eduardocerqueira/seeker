#date: 2023-04-05T16:48:41Z
#url: https://api.github.com/gists/cf537eb3421bd73a3184a604d287aa6a
#owner: https://api.github.com/users/keeth

#!/bin/bash
port=8086
cmd=/google-cloud-sdk/bin/gcloud
$cmd components update beta
$cmd beta emulators bigtable start --quiet --host-port=0.0.0.0:$port &

while true; do
  nc -z -v -w1 localhost $port >/dev/null 2>&1
  if [ $? -eq 0 ]; then
    echo "Process is listening on port $port."
    break
  else
    echo "Waiting for process to listen on port $port..."
  fi
  sleep 1
done

echo "BigTable emulator installed"

kill %1
