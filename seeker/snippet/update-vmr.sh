#date: 2022-10-13T17:23:45Z
#url: https://api.github.com/gists/fb72c5b32fef03c3f9de2e5b5e466a3c
#owner: https://api.github.com/users/lbussell

#!/bin/bash

# Extracts the dotnet source and commtis it to your $VMR repo
# Usage: ./update-vmr.sh dotnet-sdk-source.tar.gz 6.0.108

# make sure you set $VMR to your git repo:
# export VMR=/home/logan/vcs/VMR/dotnet

set -euxo pipefail

source_tarball=$1
test -f "${source_tarball}"

sdk_version=$2
month_year=$(date +"%b%Y" -d "+1 month" | sed 's/.*/\L&/') # like aug2022
new_branch_name="$dev/loganbussell/{sdk_version}-${month_year}"

dotnet_source_dirname="dotnet_source"

mkdir "${dotnet_source_dirname}"
tar -xzf "${source_tarball}" -C "${dotnet_source_dirname}"
dotnet_source_path="$(realpath ${dotnet_source_dirname})"

pushd $VMR 
    git fetch
    git checkout -b "${new_branch_name}" "release/6.0.1xx"

    # delete all contents except the .git folder
    # otherwise we won't catch deleted files in a commit
    ls | grep -v ".git" | xargs rm -rf
    cp -r "${dotnet_source_path}"/* .

    git add .
    git commit -m "Update to .NET ${sdk_version}"
popd