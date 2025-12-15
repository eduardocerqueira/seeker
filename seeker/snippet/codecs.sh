#date: 2025-12-15T17:15:06Z
#url: https://api.github.com/gists/8066b39a0989eb3dad3afa1e40ac55ec
#owner: https://api.github.com/users/kntjspr

# Free and Non-Free repos
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install ffmpeg gstreamer1-libav gstreamer1-plugins-ugly gstreamer1-plugins-bad-free --allowerasing