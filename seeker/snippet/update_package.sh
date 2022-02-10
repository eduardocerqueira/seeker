#date: 2022-02-10T16:43:00Z
#url: https://api.github.com/gists/cec1d4ecfb729ad95c788d53df8dd7f1
#owner: https://api.github.com/users/oirodolfo

# Add "test": "jest" to scripts object of node package
echo $(jq '.scripts.test="jest"' package.json) | jq . | > package.json

# Update package name
echo $(jq '.name="new-name"' package.json) | jq . | > package.json

# NOTE
# If you get this error: "pkg-script:1: file exists: package.json"
# run the following:
#
setopt clobber