#date: 2024-08-27T17:06:11Z
#url: https://api.github.com/gists/231ab631fa3f46152a2db92910d38d4b
#owner: https://api.github.com/users/thatrandomperson5

echo "Installing VSCode CLI in $PREFIX/bin"

ARCHITECTURE=$(lscpu | grep "Architecture" | tr -d " \t\n\r" | cut -c 14-) # Extract the architecture
URL=""

if ["$ARCHITECTURE" = "arm64"] || ["$ARCHITECTURE" = "aarch64"]; then
  URL="https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64"
elif ["$ARCHITECTURE" = "arm32"]; then
  URL="https://code.visualstudio.com/sha/download?build=stable&os=cli-linux-armhf"
elif ["$ARCHITECTURE" = "x64"] || ["$ARCHITECTURE" = "x86_64"] || ["$ARCHITECTURE" = "amd64"]; then
  URL="https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64"
fi

curl -sSfL "$URL" -o "$PREFIX/bin/vscode.tar.gz"

