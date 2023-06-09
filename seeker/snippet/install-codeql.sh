#date: 2023-06-09T16:53:49Z
#url: https://api.github.com/gists/2cb47738ba1a2e217a7c84cf230d222e
#owner: https://api.github.com/users/garthhumphreys

#!/bin/bash

# Check for latest release: https://github.com/github/codeql-cli-binaries/releases
_version='v2.4.1'
_arch='osx'
_zip_url="https://github.com/github/codeql-cli-binaries/releases/download/${_version}/codeql-${_arch}64.zip"
_dir='codeql-home'
_cores=2

pushd "${HOME}" || exit
# Download and extract into
curl -o "${_dir}.zip" "${_zip_url}" -L

# Unzips to codeql/
unzip "${_dir}.zip"
# Change the name
mv codeql/ "${_dir}/"

pushd "${_dir}/" || exit
# Grab the repo for the examples and query suites. You'll need these!
git clone https://github.com/github/codeql.git codeql-repo
ln -s "${HOME}/${_dir}/codeql" /usr/local/bin/codeql

echo "YAY! codeql is now installed!"
codeql --help

echo "Now run the following to create a db:"
echo "cd yourrepo/; codeql database create yourrepo-db --language=javascript"
echo "Now you can analyze!"
echo "codeql database analyze /yourrepo/yourrepo-db $HOME/codeql-home/codeql-repo/javascript/ql/src/codeql-suites/javascript-security-and-quality.qls --format=csv --output=sec-quality.csv -j $_cores"