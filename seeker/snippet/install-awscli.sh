#date: 2024-09-18T17:00:55Z
#url: https://api.github.com/gists/b7c4ed2118fe9954b85eda3d411150fb
#owner: https://api.github.com/users/salutgeek

#!/usr/bin/env bash
# see: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-source-install.html
# This script will install aws-cli libraries to /usr/local/lib/aws-cli/
# This script will install aws-cli executable to /usr/local/bin/

set -e
WORK_DIR=$(mktemp -d)

# download source package and un-tar
curl -fsSL https://awscli.amazonaws.com/awscli.tar.gz | \
	tar -xz --strip-components=1 - -C "$WORK_DIR" 

# cleanup
trap "sudo rm -rf "$WORK_DIR"" EXIT

pushd "$WORK_DIR"

# remove existing installed aws-cli
sudo rm -rf /usr/local/lib/aws-cli

# configure deps
./configure --with-download-deps

# install
make
sudo make install
popd

aws --version