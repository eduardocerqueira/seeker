#date: 2021-08-31T02:33:34Z
#url: https://api.github.com/gists/288900701f28d2857f6b6c60a1c9b704
#owner: https://api.github.com/users/gabrielsond

# tested on 2 x https://www.techpowerup.com/gpu-specs/msi-rtx-3060-ventus-2x-oc-lhr.b9132
# unraid nvidia driver community app installed and running v460.84

wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/xorg-server-1.18.3-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/xinit-1.3.4-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libxcb-1.11.1-x86_64-1.txz 
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXau-1.0.8-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXdmcp-1.1.2-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/n/nettle-3.2-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libdrm-2.4.68-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXfont-1.5.1-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/l/harfbuzz-1.2.7-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/l/freetype-2.6.3-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libfontenc-1.1.3-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libxshmfence-1.2-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/xkeyboard-config-2.17-noarch-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/xkbcomp-1.3.0-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libxkbfile-1.0.9-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/xterm-325-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXft-2.3.2-x86_64-3.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/fontconfig-2.11.1-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXaw-1.0.13-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXmu-1.1.2-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXt-1.1.5-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXinerama-1.1.3-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXpm-3.5.11-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libICE-1.0.9-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXrender-0.9.9-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXext-1.3.3-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libSM-1.2.2-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/l/gtk+2-2.24.30-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/l/atk-2.18.0-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/l/gdk-pixbuf2-2.32.3-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/l/pango-1.38.1-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXi-1.7.6-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXrandr-1.5.0-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXcursor-1.1.14-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXcomposite-0.4.4-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/l/cairo-1.14.6-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/mesa-11.2.2-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXdamage-1.1.4-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXv-1.0.10-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXfixes-5.0.2-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/x/libXxf86vm-1.1.4-x86_64-2.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/d/gdb-7.11.1-x86_64-1.txz
wget https://slackware.uk/slackware/slackware64-14.2/slackware64/d/python-2.7.11-x86_64-2.txz

upgradepkg --install-new xorg-server-1.18.3-x86_64-2.txz
upgradepkg --install-new xinit-1.3.4-x86_64-2.txz
upgradepkg --install-new libxcb-1.11.1-x86_64-1.txz 
upgradepkg --install-new libXau-1.0.8-x86_64-2.txz 
upgradepkg --install-new libXdmcp-1.1.2-x86_64-2.txz 
upgradepkg --install-new nettle-3.2-x86_64-1.txz
upgradepkg --install-new libdrm-2.4.68-x86_64-1.txz 
upgradepkg --install-new libXfont-1.5.1-x86_64-2.txz 
upgradepkg --install-new harfbuzz-1.2.7-x86_64-1.txz 
upgradepkg --install-new freetype-2.6.3-x86_64-1.txz
upgradepkg --install-new libfontenc-1.1.3-x86_64-1.txz 
upgradepkg --install-new libxshmfence-1.2-x86_64-2.txz 
upgradepkg --install-new xkeyboard-config-2.17-noarch-1.txz 
upgradepkg --install-new xkbcomp-1.3.0-x86_64-2.txz 
upgradepkg --install-new libxkbfile-1.0.9-x86_64-1.txz 
upgradepkg --install-new xterm-325-x86_64-1.txz 
upgradepkg --install-new libXft-2.3.2-x86_64-3.txz 
upgradepkg --install-new fontconfig-2.11.1-x86_64-2.txz 
upgradepkg --install-new libXaw-1.0.13-x86_64-1.txz 
upgradepkg --install-new libXmu-1.1.2-x86_64-2.txz 
upgradepkg --install-new libXt-1.1.5-x86_64-1.txz 
upgradepkg --install-new libXinerama-1.1.3-x86_64-2.txz 
upgradepkg --install-new libXpm-3.5.11-x86_64-2.txz 
upgradepkg --install-new libICE-1.0.9-x86_64-2.txz 
upgradepkg --install-new libXrender-0.9.9-x86_64-1.txz 
upgradepkg --install-new libXext-1.3.3-x86_64-2.txz 
upgradepkg --install-new libSM-1.2.2-x86_64-2.txz 
upgradepkg --install-new gtk+2-2.24.30-x86_64-1.txz 
upgradepkg --install-new atk-2.18.0-x86_64-1.txz 
upgradepkg --install-new gdk-pixbuf2-2.32.3-x86_64-1.txz 
upgradepkg --install-new pango-1.38.1-x86_64-1.txz 
upgradepkg --install-new libXi-1.7.6-x86_64-1.txz 
upgradepkg --install-new libXrandr-1.5.0-x86_64-1.txz 
upgradepkg --install-new libXcursor-1.1.14-x86_64-2.txz 
upgradepkg --install-new libXcomposite-0.4.4-x86_64-2.txz 
upgradepkg --install-new cairo-1.14.6-x86_64-2.txz 
upgradepkg --install-new mesa-11.2.2-x86_64-1.txz 
upgradepkg --install-new libXdamage-1.1.4-x86_64-2.txz 
upgradepkg --install-new libXv-1.0.10-x86_64-2.txz 
upgradepkg --install-new libXfixes-5.0.2-x86_64-1.txz 
upgradepkg --install-new libXxf86vm-1.1.4-x86_64-2.txz 
upgradepkg --install-new gdb-7.11.1-x86_64-1.txz 
upgradepkg --install-new python-2.7.11-x86_64-2.txz 

nvidia-smi -r
nvidia-smi -pm 1
# card 0 runs hotter
nvidia-smi -lgc 950 -i 0
nvidia-smi -lgc 1100 -i 1
nvidia-smi -pl 110
nvidia-smi
nvidia-xconfig --cool-bits=31 --allow-empty-initial-configuration --use-display-device=None --virtual=1920x1080 --enable-all-gpus --separate-x-screens
export DISPLAY=:0.0
xinit &
nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=2' --use-gtk2
nvidia-settings -a '[gpu:1]/GPUPowerMizerMode=2' --use-gtk2
# card 0 runs hotter
nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=2300' --use-gtk2
nvidia-settings -a '[gpu:1]/GPUMemoryTransferRateOffsetAllPerformanceLevels=2500' --use-gtk2
# card 0 runs hotter
nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels=-500' --use-gtk2
nvidia-settings -a '[gpu:1]/GPUGraphicsClockOffsetAllPerformanceLevels=-400' --use-gtk2
nvidia-smi

# Run NBMiner

# Mining Statistics (after about 30 minutes of uptime)
# GPU			Hashrate	Hashrate2	Accept1	    Reject1	Invalid1	Accept2		Reject2		Temp	Power	Fan	Core Clock	Mem Clock	Mem Util
# GeForce RTX 3060	31.31 MH/s	0.000 H/s	6		0		0		undefined	undefined	72 C	108 W	69 %	952 MHz		8451 MHz	100 %
# GeForce RTX 3060	32.65 MH/s	0.000 H/s	13		0		0		undefined	undefined	63 C	109 W	42 %	1012 MHz	8551 MHz	100 %
