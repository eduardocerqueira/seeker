#date: 2024-06-20T16:45:40Z
#url: https://api.github.com/gists/3d6060dcf99687ce4e53529f69774fa1
#owner: https://api.github.com/users/bharathibh

INSTALL_DIR="/usr/local/bin"
GECKO_URL="https://api.github.com/repos/mozilla/geckodriver/releases/latest"
TARGET_PLATFORM="linux64"

install -d -m 0755 /etc/apt/keyrings
wget -q https://packages.mozilla.org/apt/repo-signing-key.gpg -O- | tee /etc/apt/keyrings/packages.mozilla.org.asc >/dev/null
gpg -n -q --import --import-options import-show /etc/apt/keyrings/packages.mozilla.org.asc | awk '/pub/{getline; gsub(/^ +| +$/,""); if($0 == "35BAA0B33E9EB396F59CA838C0BA5CE6DC6315A3") print "\nThe key fingerprint matches ("$0").\n"; else print "\nVerification failed: the fingerprint ("$0") does not match the expected one.\n"}'
echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] https://packages.mozilla.org/apt mozilla main" | tee -a /etc/apt/sources.list.d/mozilla.list >/dev/null
echo '
Package: *
Pin: origin packages.mozilla.org
Pin-Priority: 1000
' | tee /etc/apt/preferences.d/mozilla
apt-get update && apt-get install firefox -y
# geckodriver --version

rm -rf ./geckodriver

json=$(curl -s "$GECKO_URL")
url=$(echo "$json" | tr -d '\n\t\r' | jq -r --arg TARGET_PLATFORM "$TARGET_PLATFORM" '.assets[].browser_download_url | select(contains($TARGET_PLATFORM) and endswith("gz"))')

[ -z "$url" ] && echo "Error: Couldn't find the download URL." && exit 1

curl -s -L "$url" | tar -xz
chmod +x geckodriver

[ ! -w "$INSTALL_DIR" ] && echo "Error: Permission denied." && exit 1

mv geckodriver "$INSTALL_DIR" &&
    echo "Success: $INSTALL_DIR/geckodriver"

chmod +x $INSTALL_DIR/geckodriver