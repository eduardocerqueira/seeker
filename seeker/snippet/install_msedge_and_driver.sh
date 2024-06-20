#date: 2024-06-20T16:46:41Z
#url: https://api.github.com/gists/90772b03ca3a73341ab6733c0a53394f
#owner: https://api.github.com/users/bharathibh

# Install MS Edge
curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/
sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/edge stable main" > /etc/apt/sources.list.d/microsoft-edge-dev.list'
rm microsoft.gpg
apt-get update && apt-get install microsoft-edge-stable -y
microsoft-edge --version

# Install Webdriver
curl https://msedgedriver.azureedge.net/LATEST_STABLE -o MSEDGE_DRIVER_VERSION
vi -c ":set nobomb" -c ":wq" MSEDGE_DRIVER_VERSION

edgeDriverVersion=$(echo $(cat MSEDGE_DRIVER_VERSION) | sed $'s/\r//'| sed 's/ //g')
echo "MSEDGE_DRIVER_VERSION: $edgeDriverVersion"

wget https://msedgedriver.azureedge.net/$edgeDriverVersion/edgedriver_linux64.zip
unzip -o edgedriver_linux64.zip
rm edgedriver_linux64.zip
mv -f msedgedriver /app
chmod +x /app/msedgedriver
ln -s /app/msedgedriver /usr/local/bin/msedgedriver