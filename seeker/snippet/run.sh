#date: 2022-02-23T16:51:20Z
#url: https://api.github.com/gists/190b5f7997fe895b51dae7c09ab93cb3
#owner: https://api.github.com/users/averikitsch

#!/usr/bin/env bash

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eo pipefail

# Create mount directory for service.
mkdir -p $MNT_DIR

echo "Mounting Cloud Filestore."
mount -o nolock $FILESTORE_IP_ADDRESS:/$FILE_SHARE_NAME $MNT_DIR
echo "Mounting completed."

# Run the web service on container startup.
./app/server

# Exit immediately when one of the background processes terminate.
wait -n
