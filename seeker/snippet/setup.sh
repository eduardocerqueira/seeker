#date: 2024-02-01T16:58:26Z
#url: https://api.github.com/gists/6d31e30fb03ff05039d968daa6d76740
#owner: https://api.github.com/users/Xatrekak

#!/usr/bin/bash
###############################################################################
#############################Script Setup######################################
###############################################################################
echo -e "Running script setup.\n"
sleep 2

#Disable history exspansion for our script
set +H 

# Function to show help/usage
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --amdgpu               Enable AMD GPU settings"
    echo "  --nv_unstable          Use the unstable nvidia drivers"
    echo "  --disable_nas          Copy files from NAS but don't setup back-syncing"
    echo "  --singledisplay        Enable AMD GPU settings"
}

# Initialize variables for the options
amdgpu=false
nv_unstable=false
disable_nas=false
singledisplay=false

# Use getopt to parse the options
TEMP=$(getopt -o '' --long amdgpu,nv_unstable,singledisplay,help -n 'setup.sh' -- "$@")
if [ $? != 0 ]; then
    show_help
    exit 1
fi

# Extract options and their arguments into variables.
eval set -- "$TEMP"

while true; do
    case "$1" in
        --amdgpu )
            amdgpu=true
            shift
            ;;
        --nv_unstable )
            nv_unstable=true
            shift
            ;;
        --disable_nas )
            disable_nas=true
            shift
            ;;
        --singledisplay )
            singledisplay=true
            shift
            ;;
        --help )
            show_help
            exit 0
            ;;
        -- )
            shift
            break
            ;;
        * )
            break
            ;;
    esac
done

# Now you can use the $amdgpu and $stable variables in if statements
if [ "$amdgpu" = false ]; then
    echo "Using default Nvidia configuration."
else
    echo "AMD GPU option selected."
fi

if [ "$nv_unstable" = false ]; then
    echo "Stable configuration option selected."
else
    echo "Unstable Nvidia driver selected."
fi

if [ "$disable_nas" = false ]; then
    echo "Enabling back sync to NAS."
else
    echo "Copying files from NAS but not enabling back-sync."
fi

if [ "$singledisplay" = false ]; then
    echo "Setting up shell for dual display."
else
    echo "Setting up shell for single display."
fi

###############################################################################
#############################System Configuration##############################
###############################################################################
echo -e "\nBegining with system configuration."
sleep 3

#audio setup
# `pactl list short sinks` to get the output names for sink
# `pactl list short sources` to get the input names for source
sudo dnf install -y pactl
pactl set-default-sink alsa_output.usb-Schiit_Audio_Schiit_Modi_-00.analog-stereo
pactl set-default-source alsa_input.usb-Blue_Microphones_Yeti_Stereo_Microphone_REV8-00.analog-stereo

#setup NAS
sudo dnf install -y nfs-utils
sudo mkdir /mnt/nas
sudo bash -c "echo '192.168.1.81:/mnt/tank/dustin    /mnt/nas    nfs    defaults    0 0' >> /etc/fstab"
sudo systemctl daemon-reload
sudo mount /mnt/nas

#Create symlink to libnvidia-ml.so.1 so t-rex miner can access NVML to monitor GPU
if [ "$amdgpu" = false ]; then
    sudo ln -s /usr/lib/libnvidia-ml.so.1 /usr/lib/libnvidia-ml.so
    sudo ln -s /usr/lib64/libnvidia-ml.so.1 /usr/lib64/libnvidia-ml.so
fi

#revert back to unpredicatble kernerl interface names and set grub time to 0
sudo bash -c "cat > /etc/default/grub << EOT
GRUB_DEFAULT='saved'
GRUB_DISABLE_RECOVERY='true'
GRUB_DISABLE_SUBMENU='true'
GRUB_ENABLE_BLSCFG='true'
GRUB_TERMINAL_OUTPUT='console'
GRUB_TIMEOUT='0'
GRUB_CMDLINE_LINUX_DEFAULT='quiet splash rhgb nvidia-drm.modeset=1 rd.driver.blacklist=nouveau modprobe.blacklist=nouveau'
GRUB_DISTRIBUTOR='Nobara Linux'
GRUB_CMDLINE_LINUX='nofb net.ifnames=0'
EOT"
sleep 1
sudo update-grub

#Upgrade pip
pip install --upgrade pip --quiet

echo System configuration finished.
###############################################################################
#############################Shell Configuration###############################
###############################################################################
echo Beginning with shell configuration.
sleep 1

#Setup background service
git clone https://github.com/Xatrekak/Gnome-rnd-wp
cd Gnome-rnd-wp
./install.sh --nsfw_lvl auto
cd ..
rm -rf Gnome-rnd-wp/

#Setup gnome shell behavior
gsettings set org.gnome.desktop.interface enable-hot-corners false

#Enable numlock
gsettings set org.gnome.desktop.peripherals.keyboard numlock-state true

#Setup Japanese input for Gnome via mozc and its dependencies
sudo dnf install -y mozc ibus-mozc
gsettings set org.gnome.desktop.input-sources sources "[('xkb', 'us'), ('ibus', 'mozc-on')]"

#Move lanuge switch hotkey to alt +shift
gsettings set org.gnome.desktop.wm.keybindings switch-input-source "['<Control>space']"

#Setup gnome file browser
gsettings set org.gtk.gtk4.Settings.FileChooser sort-directories-first true
gsettings set org.gnome.nautilus.preferences show-image-thumbnails 'always'
gsettings set org.gnome.nautilus.preferences show-delete-permanently true
gsettings set org.gnome.nautilus.preferences show-create-link true
gsettings set org.gnome.nautilus.preferences always-use-location-entry true
echo 'file:///mnt/nas' >> ~/.config/gtk-3.0/bookmarks
mkdir ~/Projects
echo file://$HOME/Projects >> ~/.config/gtk-3.0/bookmarks

#Setup Gnome file templates
mv ~/Templates/Text\ file ~/Templates/text_file.txt
touch ~/Templates/empty_file
touch ~/Templates/script.sh
echo -e '#!/bin/bash\n' > ~/Templates/script.sh
touch ~/Templates/py_script.py
echo -e '#!/usr/bin/python3' > ~/Templates/py_script.py

#Enable and setup system gnome extensions 
gnome-extensions enable appindicatorsupport@rgcjonas.gmail.com
gsettings set org.gnome.shell favorite-apps \
"['org.gnome.Nautilus.desktop', 'org.gnome.Terminal.desktop', 'org.gnome.gedit.desktop', 'firefox.desktop', \
'VScode.desktop', 'com.discordapp.Discord.desktop', 'steam.desktop', 'org.gnome.Software.desktop']"

#setup Gnome Weather
gsettings set org.gnome.Weather locations "[<(uint32 2, <('Alexandria', 'KDCA', true, \
[(0.67803131976116615, -1.3444998506811625)], [(0.67727215389642637, -1.3447224499739618)])>)>, \
<(uint32 2, <('Washington DC, Reagan National Airport', 'KDCA', false, [(0.67803131976116615, -1.3444998506811625)], @a(dd) [])>)>]"

#Install dependencies for gnome user extensions
sudo dnf install -y ddcutil #needed for brightness-control-using-ddcutil

#Install Gnome extension CLI installer
pip install gnome-extensions-cli

#Enable user extensions
gsettings set org.gnome.shell disable-user-extensions false

#install user gnome extensions
<<extensions
https://extensions.gnome.org/extension/307/dash-to-dock/
https://extensions.gnome.org/extension/517/caffeine/
https://extensions.gnome.org/extension/4362/fullscreen-avoider/
https://extensions.gnome.org/extension/2645/brightness-control-using-ddcutil/
https://extensions.gnome.org/extension/4228/wireless-hid/
https://extensions.gnome.org/extension/3193/blur-my-shell/
https://extensions.gnome.org/extension/841/freon/
extensions
gext install 307 517 4362 2645 4228 3193 841

#setup gnome user extensions
#dash-to-doc
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock multi-monitor true
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock show-mounts false
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock show-mounts-network false
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock show-show-apps-button false
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock show-trash false
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock disable-overview-on-startup true
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock apply-custom-theme false
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock custom-theme-shrink true
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock transparency-mode 'FIXED'
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock dash-max-icon-size 64
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock autohide true
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock custom-background-color true
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock background-color 'rgb(145,65,172)'
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock background-opacity 0.30
gsettings --schemadir ~/.local/share/gnome-shell/extensions/dash-to-dock@micxgx.gmail.com/schemas/ set org.gnome.shell.extensions.dash-to-dock intellihide-mode 'ALL_WINDOWS'
#caffeine
gsettings --schemadir ~/.local/share/gnome-shell/extensions/caffeine\@patapon.info/schemas/ set org.gnome.shell.extensions.caffeine show-indicator 'always'
#freon
gsettings --schemadir ~/.local/share/gnome-shell/extensions/freon@UshakovVasilii_Github.yahoo.com/schemas set org.gnome.shell.extensions.freon hot-sensors \
"['__average__', '__max__', 'NVIDIA GeForce RTX 3090', 'T_Sensor']"
gsettings --schemadir ~/.local/share/gnome-shell/extensions/freon@UshakovVasilii_Github.yahoo.com/schemas set org.gnome.shell.extensions.freon use-gpu-nvidia true

#Theme setup
echo 'export GTK_THEME=Dracula' >> ~/.bash_profile
mkdir ~/.themes
wget -O /tmp/Dracula.zip https://github.com/dracula/gtk/archive/master.zip
unzip /tmp/Dracula.zip -d ~/.themes
mv ~/.themes/gtk-master/ ~/.themes/Dracula
gsettings set org.gnome.desktop.interface gtk-theme "Dracula"
gsettings set org.gnome.desktop.wm.preferences theme "Dracula"

#Alias setup
cat >> ~/.bashrc << EOT

# User defined alises
alias FAHControl='python2 /usr/bin/FAHControl'
alias mnt="mount | awk -F' ' '{ printf \"%s\\\t%s\\\n\", \\\$1, \\\$3; }' | column -t | egrep '^(/dev/|.*nas)' | sort"
alias gh='history|grep'
alias cpv='rsync -ah --info=progress2'
alias reboot='sudo reboot now'
alias reload='sudo reboot now'
EOT

#Setup monitors
if [ "$singledisplay" = false ]; then
git clone "https://gitlab.com/Oschowa/gnome-randr"
mv gnome-randr /tmp/gnome-randr
chmod +x /tmp/gnome-randr/gnome-randr.py
/tmp/gnome-randr/gnome-randr.py --output DP-1 --mode 2560x1440 --rate 120
/tmp/gnome-randr/gnome-randr.py --output HDMI-1 --mode 2560x1440 --rate 120 --primary
fi

echo Shell configuration finished.
###############################################################################
#############################App Configuration#################################
###############################################################################
echo Beginning with app configuration.
sleep 1

#Install and setup lsyncd for easy rysnc to nas
if [ "$disable_nas" = false ]; then
sudo dnf install -y lsyncd
mkdir -p ~/.local/state/lsyncd/
touch ~/.local/state/lsyncd/lsyncd.log
touch ~/.local/state/lsyncd/lsyncd.status
sudo bash -c "cat > /etc/lsyncd.conf << EOT
settings {
   logfile    = \"$HOME/.local/state/lsyncd/lsyncd.log\",
   statusFile = \"$HOME/.local/state/lsyncd/lsyncd.status\",
   nodaemon   = false,
}

sync {
    default.rsync,
    source    = \"$HOME/.mozilla\",
    target    = \"/mnt/nas/firefox/.mozilla\",
    rsync     = {
        archive  = true,    -- enables recursion and preserves almost everything
        compress = false,    -- compresses data during the transfer
        _extra   = {\"--delete\"} -- includes deletion of files in the target
    }
}
EOT"
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/lsyncd.service << EOT
[Unit]
Description=Live Syncing (Mirror) Daemon
Documentation=man:lsyncd(1) file://usr/share/doc/lsyncd/README.md https://axkibe.github.io/lsyncd/
After=network.target

[Service]
Type=forking
Nice=19
EnvironmentFile=-/etc/sysconfig/lsyncd
ExecStart=/usr/bin/lsyncd -nodaemon \$LSYNCD_OPTIONS
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
SuccessExitStatus=143

[Install]
WantedBy=default.target
EOT
fi


#Setup firefox.
#link .mozzila on nas and profile
echo Copying firefox config from NAS to host, this may take a while.
echo "Closing all isntances of firefox. Do not open until finished"
pkill firefox
sleep 1
rm -rf ~/.mozilla
sleep 2
rsync -ah --info=progress2 /mnt/nas/firefox/.mozilla ~/
# Check if the directory exists so we don't wipe out the backup
if [ "$disable_nas" = false ]; then
if [ -d ~/.mozilla ]; then
    systemctl start --user lsyncd
    systemctl enable --user lsyncd
else
    echo "Copying mozilla directory from the NAS has failed."
fi
fi

#install apps
#Flatpaks
#Fleatseal
echo Installing Flatseal
sudo flatpak install flathub com.github.tchx84.Flatseal --noninteractive --system
#Anki
echo Installing Anki
flatpak install flathub net.ankiweb.Anki --noninteractive --user
flatpak override --user net.ankiweb.Anki --filesystem=host --socket=wayland --nosocket=x11
echo "export ANKI_WAYLAND=1" >> ~/.bash_profile
#Eye of gnome
echo Installing Eye of Gnome
flatpak install flathub org.gnome.eog --noninteractive --user
#Discord
echo Installing Discord
flatpak install flathub com.discordapp.Discord --noninteractive --user
#ExtensionManager
echo Installing ExtensionManager
flatpak install flathub com.mattjakeman.ExtensionManager --noninteractive --user
#onlyoffice
echo Installing onlyoffice
flatpak install flathub org.onlyoffice.desktopeditors --noninteractive --user
#Filezilla
echo Installing Filezilla
flatpak install flathub org.filezillaproject.Filezilla --noninteractive --user
#Edge
echo Installing Edge
flatpak install flathub com.microsoft.Edge --noninteractive --user
#qBittorrent
echo Installing qBittorrent
flatpak install flathub org.qbittorrent.qBittorrent --noninteractive --user
#visualstudio.code
echo Installing visualstudio.code
flatpak install flathub com.visualstudio.code --noninteractive --user
##Set VSCode to use host terminal
mkdir -p ~/.var/app/com.visualstudio.code/config/Code/User/
touch ~/.var/app/com.visualstudio.code/config/Code/User/settings.json
cat > ~/.var/app/com.visualstudio.code/config/Code/User/settings.json << EOT
{
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.profiles.linux": {
      "bash": {
        "path": "host-spawn",
        "args": ["bash"]
      }
    },
    "security.workspace.trust.enabled": false
  }
EOT
##Force code to run native wayland.
flatpak override --user com.visualstudio.code --socket=wayland --socket=fallback-x11 --nosocket=x11
touch ~/.local/share/applications/VScode.desktop
cat > ~/.local/share/applications/VScode.desktop << EOT
[Desktop Entry]
Name=Visual Studio Code
Comment=Code Editing. Redefined.
GenericName=Text Editor
Exec=/usr/bin/flatpak run com.visualstudio.code --enable-features=UseOzonePlatform,WaylandWindowDecorations --ozone-platform=wayland
Icon=com.visualstudio.code
Type=Application
StartupNotify=true
StartupWMClass=Code
Categories=TextEditor;Development;IDE;
MimeType=text/plain;inode/directory;application/x-code-workspace;
Actions=new-empty-window;
Keywords=vscode;
X-Flatpak-Tags=proprietary;
X-Flatpak=com.visualstudio.code

[Desktop Action new-empty-window]
Name=New Empty Window
Exec=/usr/bin/flatpak run com.visualstudio.code --enable-features=UseOzonePlatform,WaylandWindowDecorations --ozone-platform=wayland --new-window
Icon=com.visualstudio.code
EOT

#install GNS3
echo Installing gns3-gui
pip install gns3-gui --quiet

#setup and install input-leap
echo Installing input-leap
sudo firewall-cmd --permanent --add-port=24800/tcp
sudo firewall-cmd --reload
sudo dnf4 copr enable -y ofourdan/input-leap-ei-enabled
sudo dnf4 install -y input-leap --repo copr:copr.fedorainfracloud.org:ofourdan:input-leap-ei-enabled
sudo dnf4 reinstall -y libportal --repo copr:copr.fedorainfracloud.org:ofourdan:input-leap-ei-enabled

#Install FAH
sudo dnf install -y pygtk2
mkdir /tmp/fah
wget -O /tmp/fah/fahclient.rpm https://download.foldingathome.org/releases/public/release/fahclient/centos-6.7-64bit/v7.6/latest.rpm
wget -O /tmp/fah/fahcontrol.rpm https://download.foldingathome.org/releases/public/release/fahcontrol/centos-6.7-64bit/v7.6/latest.noarch.rpm
wget -O /tmp/fah/fahviewer.rpm https://download.foldingathome.org/releases/public/release/fahviewer/centos-6.7-64bit/v7.6/latest.rpm
sudo dnf install -y  /tmp/fah/fahclient.rpm
sudo dnf install -y  /tmp/fah/fahcontrol.rpm
sudo dnf install -y  /tmp/fah/fahviewer.rpm
#Setup FAH
sudo mkdir -p /etc/fahclient/
sudo touch /etc/fahclient/config.xml
sudo bash -c "cat > /etc/fahclient/config.xml << EOT
<config>
  <!-- Network -->
  <proxy v=':8080'/>

  <!-- Slot Control -->
  <power v='full'/>

  <!-- User Information -->
  <passkey v='3bd1d47c043d68653bd1d47c043d6865'/>
  <team v='223518'/>
  <user v='xatrekak'/>

  <!-- Folding Slots -->
  <slot id='0' type='GPU'>
    <pci-bus v='9'/>
    <pci-slot v='0'/>
  </slot>
</config>
EOT"

#Remove unused apps
sudo dnf remove -y libreoffice*
sudo dnf remove -y loupe

echo App configuration finished.
###############################################################################
#############################Final Configuration###############################
###############################################################################
echo Beginning with Final configuration.
sleep 1

#Setup and install xbox controller dependencies.
sudo usermod -a -G pkg-build $USER
sudo dnf install -y "dnf5-command(builddep)"

#Setup restart script to install xbox controller on next reboot.
mkdir -p ~/.config/autostart
touch ~/.config/autostart/setup.desktop
touch ~/.config/autostart/setup.sh
chmod +x ~/.config/autostart/setup.sh
cat > ~/.config/autostart/setup.desktop << EOT
[Desktop Entry]
Type=Application
Name=startup script
Exec=$HOME/.config/autostart/setup.sh
#X-GNOME-Autostart-enabled=true
EOT
cat > ~/.config/autostart/setup.sh << EOT
sleep 60
nobara-controller-config
rm ~/.config/autostart/setup.desktop
rm ~/.config/autostart/setup.sh
EOT

#Move to Nvidia new feature branch
if [ "$nv_unstable" = true ]; then
sudo dnf update nobara-repos --refresh
sudo dnf4 config-manager --set-enabled nobara-nvidia-new-feature-39
sudo dnf update -y --refresh
sudo akmods
sudo nobara-sync
fi

#Reboot the system.
echo Configuration finished. Rebooing in: 
echo 3
sleep 1
echo 2
sleep 1
echo 1
sleep 1
sudo reboot now
