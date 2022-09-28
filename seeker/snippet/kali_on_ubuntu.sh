#date: 2022-09-28T17:26:18Z
#url: https://api.github.com/gists/f6b41e529d46b6ca67adb028e613ac8d
#owner: https://api.github.com/users/nathan-rabet

# Adding Kali Linux Repository <devel@kali.org> key
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/kali.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys ED444FF07D8D0BF6

# Adding Kali repository
echo "deb [signed-by=/usr/share/keyrings/kali.gpg] http://http.kali.org/kali kali-rolling main contrib non-free" | sudo tee /etc/apt/sources.list.d/kali.list

# Updating the apt cache
sudo apt update