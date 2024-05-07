#date: 2024-05-07T16:46:14Z
#url: https://api.github.com/gists/fb0bd695d3b957f2d5ccd0b7c2e4c645
#owner: https://api.github.com/users/darjeelingsteve

# See https://darjeelingsteve.com/articles/Notarising-Swift-Package-Development-Tools-for-Distribution.html for a full description

printHelp() {
    read -r -d '' HELP << EOM
Usage:
    build-and-sign.sh <tool-name> <version-number>
EOM
    >&2 echo "$HELP"
}

if [ $# -ne 2 ]; then
    printHelp
    exit 1
fi

tool_name=$1
version_number=$2

# Build the package
xcrun swift build -c release --arch arm64 --arch x86_64

# Remove existing build artefact (if any)
rm $tool_name

# Copy the built binary to the current directory
ditto .build/apple/Products/Release/$tool_name .

# Codesign the binary. `-o runtime` specifies the hardened runtime
codesign -o runtime -s "<developer-id-identity>" $tool_name

# Zip the signed binary
ditto -c -k $tool_name $tool_name-$version_number.zip

# Upload the signed zip file to the notary service
xcrun notarytool submit $tool_name-$version_number.zip --keychain-profile "NOTARY_PASSWORD" --wait
