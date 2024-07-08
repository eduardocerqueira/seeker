#date: 2024-07-08T16:55:24Z
#url: https://api.github.com/gists/da286b192aa521e00fe2c63214687049
#owner: https://api.github.com/users/MaxenceLebrunDEV

sudo apt update && sudo apt upgrade -y
sudo apt install qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils virtinst libvirt-daemon cockpit cockpit-* neofetch nano curl htop -y
sudo bash -c '> /etc/motd'
sudo bash -c $'echo "neofetch" >> /etc/profile.d/mymotd.sh && chmod +x /etc/profile.d/mymotd.sh'
sudo systemctl start cockpit
sudo systemctl enable cockpit
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && 
sudo dpkg -i cloudflared.deb && sudo cloudflared service install eyJhIjoiZGFjZWM3OWY3NTNkNWVmN2U2MDAyZmY3YmYxYTQ4YzciLCJ0IjoiMTcwNThiYjgtOTk2YS00OWU3LTgxYjQtODE4NWMxZjAyODY5IiwicyI6IlpqWTROR1ZpTkRJdE5EWXpaaTAwTVRGaUxXRXpObVV0WVRVMVkyVmxOMkk1TW1VMyJ9
cd /home/
echo '# Mise en place du serveur terminé'