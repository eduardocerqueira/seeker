#date: 2022-02-21T16:52:47Z
#url: https://api.github.com/gists/739b0a44018ed61404495fe269b46d90
#owner: https://api.github.com/users/kamakazix

sudo dnf update
sudo dnf install --nogpgcheck https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
sudo dnf install --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-8.noarch.rpm 
sudo dnf config-manager --enable powertools
sudo dnf update

sudo dnf install nano bash-completion plasma-desktop plasma-nm sddm sddm-breeze sddm-kcm konsole5 dolphin okular ark gwenview spectacle kwrite kcalc kscreen
sudo dnf groups install Fonts

sudo systemctl enable sddm
sudo systemctl set-default graphical.target