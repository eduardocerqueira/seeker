#date: 2024-05-17T16:54:36Z
#url: https://api.github.com/gists/cf16552f20c3ea15c1ceab5472ebe586
#owner: https://api.github.com/users/gustavonovaes

## Debian 12

#!/bin/bash

## Create a steam user
useradd -m steam
passwd steam
usermod -a -G sudo steam

## Sign in
sudo -u steam -s

## Install Steam dependencies 
sudo apt update
sudo apt install software-properties-common
sudo apt-add-repository non-free 
sudo dpkg --add-architecture i386 
sudo apt update
sudo apt install steamcmd

## Install WineHQ + xvfb 
sudo mkdir -pm755 /etc/apt/keyrings
sudo wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key
sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/debian/dists/bookworm/winehq-bookworm.sources
sudo apt install --install-recommends winehq-staging xvfb 


## Install game
/usr/games/steamcmd +@sSteamCmdForcePlatformType windows \  
  +force_install_dir /home/steam/abiotic-factor \
  +login anonymous \
  +app_update 2857200 \
  +quit


## Expose ports 7777 and 27015 permanently
sudo apt install iptables-persistent 
sudo iptables -A INPUT -p tcp --dport 7777 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 7777 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 27015 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 27015 -j ACCEPT
sudo iptables-save | sudo tee /etc/iptables/rules.v4
sudo ip6tables-save | sudo tee /etc/iptables/rules.v6


## Create a service for server /etc/systemd/system/abiotic.service 
[Unit]
Description=Abiotic Factor Server
Wants=network-online.target
After=network-online.target
 
[Service]
User=steam
Group=steam
WorkingDirectory=/home/steam/abiotic-factor/
ExecStartPre=/usr/games/steamcmd +@sSteamCmdForcePlatformType windows +force_install_dir /home/steam/abiotic-factor +login anonymous +app_update 2857200 +quit
ExecStart=/home/steam/abiotic-factor/run-server.sh
Restart=always

[Install]
WantedBy=multi-user.target

## Setup service 
sudo systemctl daemon-reload
sudo systemctl enable abiotic


## Init script run-server.sh 
#!/bin/bash
xvfb-run wine64 /home/steam/abiotic-factor/AbioticFactor/Binaries/Win64/AbioticFactorServer-Win64-Shipping.exe -log -newconole -useperfthreads -NoAsyncLoadingThread -MaxServerPlayers= "**********"=7777 -QUERYPORT=27015 -tcp -ServerPassword=biribelo -SteamServerName="Ameagans" -WorldSaveName="Ameagans"


## Start service
sudo systemctl start abiotic

## Sandbox setup example SandboxSettings.ini
[SandboxSettings]
ItemDurabilityMultiplier=0.1
DeathPenalties=0
SinkRefillRate=2.0
LootRespawnEnabled=True
ItemStackSizeMultiplier=10.0
GlobalRecipeUnlocks=True
EnemySpawnRate=3
DamageToAlliesMultiplier=0.0
DurabilityLossOnDeathMultiplier=0.0

## Optimize system sysctl.conf
https://gist.github.com/PSJoshi/6ac857239be01f73747a