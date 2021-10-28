#date: 2021-10-28T17:08:08Z
#url: https://api.github.com/gists/a5954777f3f58dff953f6720b303b8f1
#owner: https://api.github.com/users/Qetlin

# fix owner of files and folders recursively
sudo chown -vR $(whoami) /usr/local /opt/homebrew-cask /Library/Caches/Homebrew

# fix read/write permission of files and folders recursively
chmod -vR ug+rw /usr/local /opt/homebrew-cask /Library/Caches/Homebrew

# fix execute permission of folders recursively
find /usr/local /opt/homebrew-cask /Library/Caches/Homebrew -type d -exec chmod -v ug+x {} +