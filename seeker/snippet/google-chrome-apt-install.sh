#date: 2025-08-07T17:01:48Z
#url: https://api.github.com/gists/69d9b2204feb38d7bd9ba0f535b49882
#owner: https://api.github.com/users/globalpressinc

# Installing google chrome via APT (sadly no SNAP available from Google, like there is for Chromium)

# Setup the Google signer and repo
curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | sudo gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list

# Install
sudo apt update
sudo apt install google-chrome-stable

# Credit: This updated guide is based on an original gist by @twistedpair.