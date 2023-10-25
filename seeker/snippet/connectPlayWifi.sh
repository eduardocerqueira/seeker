#date: 2023-10-25T16:37:11Z
#url: https://api.github.com/gists/2c18668c8e4520d3db4adcc35f102f8b
#owner: https://api.github.com/users/Grippy98

#ONLY RUN ONCE
#RUN WITH SUDO

SSID=YOUR_NETWORK_NAME
PASS= "**********"


wpa_cli add_network
sudo wpa_cli set_network 1 ssid '"$SSID"'
wpa_cli set_network 1 psk '"$PASS"'
wpa_cli enable_network 1
sleep 5
ifconfig wlan0