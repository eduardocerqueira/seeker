#date: 2024-09-30T17:09:33Z
#url: https://api.github.com/gists/be078308e3fb322c4a649cdfd8444544
#owner: https://api.github.com/users/cilegordev

cd ~
clear
echo -e "\e[32mwellcome - xfce4 installer! \e[0m"
sleep 2
sudo timedatectl set-timezone Asia/Jakarta
echo "$(whoami) ALL=(ALL:ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/$(whoami)
cd /tmp
git clone https://github.com/cilegordev/azurelinux-repo && cd azurelinux-repo && sudo rm -rf /etc/yum.repos.d/* && sudo mv *.repo /etc/yum.repos.d && cd /tmp
#dependencies-required
sudo dnf -y install libndp* polkit* ppp* jansson* libpsl* libedit* newt* libsecret* iso-codes* nss* nspr* mobile-broadband-provider-info* adwaita* alsa* asciidoc* cairo* dbus* dejavu* desktop-file-utils* drm* flac* fontconfig* fribidi* gdbm* gdk* gnome* gnutls* gobject-introspection* graphene* gst* gtk* htop itstool* intltool* libICE* libSM* libX* libXtst* libburn* libcanberra* libdbus* libdvd* libexif* libgcrypt* libgudev* libisofs* libjpeg* libltdl* libnotify* libogg* libpng* librs* libsndfile* libsoup* libva* libvorbis* libvp* libvte* libxf* libxk* lz* mesa* meson* nano nasm* pam* pcre2* perl-XML-Parser* pulseaudio* pygobject* sound* unzip upower* vala* vte* vulkan* xcb* xcu* xk* xorg* xterm* zsh --skip-broken && cd /tmp
sudo ln -s /usr/bin/gcc /usr/bin/c99
#xfce4-component!
wget https://archive.xfce.org/src/xfce/libxfce4util/4.19/libxfce4util-4.19.3.tar.bz2 && tar -xf libxfce4util-4.19.3.tar.bz2 && cd libxfce4util-4.19.3 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/xfce/xfconf/4.19/xfconf-4.19.2.tar.bz2 && tar -xf xfconf-4.19.2.tar.bz2 && cd xfconf-4.19.2 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/xfce/libxfce4ui/4.19/libxfce4ui-4.19.5.tar.bz2 && tar -xf libxfce4ui-4.19.5.tar.bz2 && cd libxfce4ui-4.19.5 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/xfce/exo/4.19/exo-4.19.0.tar.bz2 && tar -xf exo-4.19.0.tar.bz2 && cd exo-4.19.0 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/xfce/garcon/4.19/garcon-4.19.1.tar.bz2 && tar -xf garcon-4.19.1.tar.bz2 && cd garcon-4.19.1 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://download.gnome.org/sources/libwnck/43/libwnck-43.0.tar.xz && tar -xf libwnck-43.0.tar.xz && cd libwnck-43.0 && mkdir build && cd build && meson setup .. --prefix=/usr --buildtype=release && sudo ninja install && cd /tmp
wget https://archive.xfce.org/src/xfce/libxfce4windowing/4.19/libxfce4windowing-4.19.3.tar.bz2 && tar -xf libxfce4windowing-4.19.3.tar.bz2 && cd libxfce4windowing-4.19.3 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/xfce/xfce4-panel/4.19/xfce4-panel-4.19.4.tar.bz2 && tar -xf xfce4-panel-4.19.4.tar.bz2 && cd xfce4-panel-4.19.4 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp 
wget https://download.gnome.org/sources/gsettings-desktop-schemas/46/gsettings-desktop-schemas-46.1.tar.xz && tar -xf gsettings-desktop-schemas-46.1.tar.xz && cd gsettings-desktop-schemas-46.1 && sed -i -r 's:"(/system):"/org/gnome\1:g' schemas/*.in && mkdir build && cd build && meson setup .. --prefix=/usr && sudo ninja install && cd /tmp
wget https://archive.xfce.org/src/xfce/thunar/4.19/thunar-4.19.3.tar.bz2 && tar -xf thunar-4.19.3.tar.bz2 && cd thunar-4.19.3 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && sudo ldconfig && cd /tmp
wget https://archive.xfce.org/src/xfce/thunar-volman/4.18/thunar-volman-4.18.0.tar.bz2 && tar -xf thunar-volman-4.18.0.tar.bz2 && cd thunar-volman-4.18.0 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/xfce/tumbler/4.19/tumbler-4.19.1.tar.bz2 && tar -xf tumbler-4.19.1.tar.bz2 && cd tumbler-4.19.1 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp 
wget https://archive.xfce.org/src/xfce/xfce4-appfinder/4.19/xfce4-appfinder-4.19.2.tar.bz2 && tar -xf xfce4-appfinder-4.19.2.tar.bz2 && cd xfce4-appfinder-4.19.2 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/xfce/xfce4-power-manager/4.19/xfce4-power-manager-4.19.3.tar.bz2 && tar -xf xfce4-power-manager-4.19.3.tar.bz2 && cd xfce4-power-manager-4.19.3 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/xfce/xfce4-settings/4.19/xfce4-settings-4.19.2.tar.bz2 && tar -xf xfce4-settings-4.19.2.tar.bz2 && cd xfce4-settings-4.19.2 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/xfce/xfdesktop/4.19/xfdesktop-4.19.2.tar.bz2 && tar -xf xfdesktop-4.19.2.tar.bz2 && cd xfdesktop-4.19.2 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/xfce/xfwm4/4.18/xfwm4-4.18.0.tar.bz2 && tar -xf xfwm4-4.18.0.tar.bz2 && cd xfwm4-4.18.0 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/xfce/xfce4-session/4.19/xfce4-session-4.19.2.tar.bz2 && tar -xf xfce4-session-4.19.2.tar.bz2 && cd xfce4-session-4.19.2 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
#xfce4-apps!
wget https://download.gnome.org/sources/gtksourceview/4.8/gtksourceview-4.8.4.tar.xz && tar -xf gtksourceview-4.8.4.tar.xz && cd gtksourceview-4.8.4 && mkdir build && cd build && meson setup .. --prefix=/usr && sudo ninja install && cd /tmp
wget https://archive.xfce.org/src/apps/mousepad/0.6/mousepad-0.6.2.tar.bz2 && tar -xf mousepad-0.6.2.tar.bz2 && cd mousepad-0.6.2 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make -k install && sudo ldconfig && cd /tmp
wget https://gstreamer.freedesktop.org/src/gstreamer/gstreamer-1.24.6.tar.xz && tar -xf gstreamer-1.24.6.tar.xz && cd gstreamer-1.24.6 && mkdir build && cd build && meson setup .. --prefix=/usr --buildtype=release -D gst_debug=false && sudo ninja install  && cd /tmp
wget https://gstreamer.freedesktop.org/src/gst-plugins-base/gst-plugins-base-1.24.6.tar.xz && tar -xf gst-plugins-base-1.24.6.tar.xz && cd gst-plugins-base-1.24.6 && mkdir build && cd build && meson setup .. --prefix=/usr --buildtype=release --wrap-mode=nodownload && sudo ninja install && cd /tmp
wget https://gstreamer.freedesktop.org/src/gst-plugins-good/gst-plugins-good-1.24.6.tar.xz && tar -xf gst-plugins-good-1.24.6.tar.xz && cd gst-plugins-good-1.24.6 && mkdir build && cd build && meson setup .. --prefix=/usr --buildtype=release && sudo ninja install && cd /tmp
wget https://gstreamer.freedesktop.org/src/gst-plugins-bad/gst-plugins-bad-1.24.6.tar.xz && tar -xf gst-plugins-bad-1.24.6.tar.xz && cd gst-plugins-bad-1.24.6 && mkdir build && cd build && meson setup .. --prefix=/usr --buildtype=release -D gpl=enabled && sudo ninja install && cd /tmp
wget https://download.gnome.org/sources/gtk/4.15/gtk-4.15.4.tar.xz && tar -xf gtk-4.15.4.tar.xz && cd gtk-4.15.4 && mkdir build && cd build && meson setup --prefix=/usr --buildtype=release -D broadway-backend=true -D introspection=enabled -D vulkan=disabled && sudo ninja install && cd /tmp
wget https://archive.xfce.org/src/apps/xfce4-terminal/1.1/xfce4-terminal-1.1.3.tar.bz2 && tar -xf xfce4-terminal-1.1.3.tar.bz2 && cd xfce4-terminal-1.1.3 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && sudo ldconfig && cd /tmp
wget https://archive.xfce.org/src/apps/xfce4-taskmanager/1.5/xfce4-taskmanager-1.5.7.tar.bz2 && tar -xf xfce4-taskmanager-1.5.7.tar.bz2 && cd xfce4-taskmanager-1.5.7 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && sudo ldconfig && cd /tmp
wget https://archive.xfce.org/src/apps/parole/4.18/parole-4.18.1.tar.bz2 && tar -xf parole-4.18.1.tar.bz2 && cd parole-4.18.1 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && sudo ldconfig && cd /tmp
wget https://archive.xfce.org/src/apps/xfburn/0.7/xfburn-0.7.1.tar.bz2 && tar -xf xfburn-0.7.1.tar.bz2 && cd xfburn-0.7.1 && ./configure --prefix=/usr --disable-static && sudo make install && sudo ldconfig && cd /tmp
wget https://archive.xfce.org/src/apps/ristretto/0.13/ristretto-0.13.2.tar.bz2 && tar -xf ristretto-0.13.2.tar.bz2 && cd ristretto-0.13.2 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && sudo ldconfig && cd /tmp
wget https://archive.xfce.org/src/xfce/xfce4-dev-tools/4.19/xfce4-dev-tools-4.19.1.tar.bz2 && tar -xf xfce4-dev-tools-4.19.1.tar.bz2 && cd xfce4-dev-tools-4.19.1 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install && cd /tmp
wget https://archive.xfce.org/src/apps/xfce4-notifyd/0.9/xfce4-notifyd-0.9.4.tar.bz2 && tar -xf xfce4-notifyd-0.9.4.tar.bz2 && cd xfce4-notifyd-0.9.4 && ./configure --prefix=/usr --sysconfdir=/etc && sudo make install ldconfig && cd /tmp
wget https://archive.xfce.org/src/panel-plugins/xfce4-pulseaudio-plugin/0.4/xfce4-pulseaudio-plugin-0.4.8.tar.bz2 && tar -xf xfce4-pulseaudio-plugin-0.4.8.tar.bz2 && cd xfce4-pulseaudio-plugin-0.4.8 && ./configure --prefix=/usr && sudo make install && cd /tmp
wget https://download.gnome.org/sources/NetworkManager/1.51/NetworkManager-1.51.1.tar.xz && tar -xvf NetworkManager-1.51.1.tar.xz && cd NetworkManager-1.51.1 && mkdir build && cd build && meson setup .. --prefix=/usr -D libaudit=no -D modem_manager=false && sudo ninja install && cd /tmp
wget https://download.gnome.org/sources/libnma/1.10/libnma-1.10.6.tar.xz && tar -xvf libnma-1.10.6.tar.xz && cd libnma-1.10.6 && mkdir build && cd build && meson setup .. --prefix=/usr -D gcr=false && sudo ninja install && cd /tmp
wget https://download.gnome.org/sources/network-manager-applet/1.36/network-manager-applet-1.36.0.tar.xz && tar -xvf network-manager-applet-1.36.0.tar.xz && cd network-manager-applet-1.36.0 && mkdir build && cd build && meson setup .. --prefix=/usr -D appindicator=no -D wwan=false && sudo ninja install && cd /tmp
wget https://ftp.mozilla.org/pub/firefox/releases/130.0/linux-x86_64/id/firefox-130.0.tar.bz2 && tar -xvf firefox-130.0.tar.bz2 && sudo mv -v firefox /opt && sudo ln -s /opt/firefox/firefox /bin && sudo ln -s /opt/firefox/firefox-bin /bin/mozilla-firefox
sudo mkdir -p /opt/gimp-2.10 && sudo wget -O /opt/gimp-2.10/gimp https://github.com/aferrero2707/gimp-appimage/releases/download/continuous/GIMP_AppImage-git-2.10.25-20210610-withplugins-x86_64.AppImage && sudo chmod +x /opt/gimp-2.10/gimp && sudo ln -s /opt/gimp-2.10/gimp /bin/gimp
wget https://kartolo.sby.datautama.net.id/tdf/libreoffice/stable/24.8.2/rpm/x86_64/LibreOffice_24.8.2_Linux_x86-64_rpm.tar.gz && gunzip LibreOffice_24.8.2_Linux_x86-64_rpm.tar.gz && tar -xvf LibreOffice_24.8.2_Linux_x86-64_rpm.tar && cd LibreOffice_24.8.2.1_Linux_x86-64_rpm/RPMS && sudo rpm -i *.rpm && cd /tmp
wget https://vscode.download.prss.microsoft.com/dbazure/download/stable/38c31bc77e0dd6ae88a4e9cc93428cc27a56ba40/code-1.93.1-1726079369.el8.x86_64.rpm && sudo rpm -i code-1.93.1-1726079369.el8.x86_64.rpm
git clone https://github.com/cilegordev/neofetch/ && cd neofetch && sudo make install && cd /tmp
git clone https://github.com/cilegordev/Flat-Adwaita && cd Flat-Adwaita && sudo mv Adwaita-* /usr/share/themes && sudo mv Flat-ZOMG-* /usr/share/icons && cd /tmp
wget https://github.com/tokotype/PlusJakartaSans/releases/download/2.7.1/PlusJakartaSans-2.7.1.zip && unzip PlusJakartaSans-2.7.1.zip && rm -rf __MACOSX && cd PlusJakartaSans-2.7.1/ttf && sudo mkdir /usr/share/fonts/plus-jakarta-sans-fonts && sudo mv * /usr/share/fonts/plus-jakarta-sans-fonts && cd ~
echo -e "XTerm*mainMenu: true \nXTerm*ToolBar: true \nXTerm*Background: black \nXTerm*Foreground: white \nXTerm*faceName: Segoe-UI:size=11 \nsudo rm -rf /efi" | tee -a .Xdefaults
echo -e "nm-applet & \npulseaudio --start \nexec dbus-launch startxfce4" | tee -a .xinitrc
cd ~ && wget https://raw.githubusercontent.com/cilegordev/Azure-Linux-WM/refs/heads/Azure-Linux/.zshrc
chsh -s $(which zsh)
sudo rm -rf /etc/profile.d/debuginfod.sh
sudo systemctl enable NetworkManager.service
sudo systemctl start NetworkManager.service
echo -e "\e[31mwarning - cleaning installer! \e[0m"
sleep 2
clear
echo -e "\e[33mdone! \e[0m"
sleep 2
startx &> ~/.x-session-errors