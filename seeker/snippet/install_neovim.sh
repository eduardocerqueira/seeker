#date: 2022-07-07T17:10:35Z
#url: https://api.github.com/gists/590468b87a21481a68fdcf1093a90b45
#owner: https://api.github.com/users/JeffLabonte

# Download latest NeoVim .deb package from GitHub using GH API
wget -o nvim.deb \
    $(curl -s 'https://api.github.com/repos/neovim/neovim/releases/latest' \
    | jq -r '.assets|.[]|select(.content_type == "application/x-debian-package")|.browser_download_url')

# Install .deb
sudo dpkg -i nvim.deb

# Clean up
rm -f nvim.deb
