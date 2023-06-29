#date: 2023-06-29T17:04:51Z
#url: https://api.github.com/gists/cd70859b4e20775c3b6a38c91ac81588
#owner: https://api.github.com/users/esynr3z

sudo dpkg --add-architecture i386
sudo apt update
sudo apt install -y libc6:i386 libxtst6:i386 libncurses5:i386 libxft2:i386 libstdc++6:i386 libc6-dev-i386 lib32z1 libqt5xml5 liblzma-dev
wget https://download.altera.com/akdlm/software/acdsinst/20.1std/711/ib_installers/ModelSimSetup-20.1.0.711-linux.run
chmod +x ModelSimSetup-20.1.0.711-linux.run
./ModelSimSetup-20.1.0.711-linux.run --mode unattended --accept_eula 1 --installdir $HOME/ModelSim-20.1.0 --unattendedmodeui none
# Here you need to add "$HOME/ModelSim-20.1.0/modelsim_ase/bin" to your PATH