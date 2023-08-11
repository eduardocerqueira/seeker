#date: 2023-08-11T17:05:10Z
#url: https://api.github.com/gists/8c7082d9b65a3b3890c3b865c861f773
#owner: https://api.github.com/users/docsolarstone

# Open Android APP Termux-X11 client
export DISPLAY=:1

PULSE_SERVER=tcp:127.0.0.1:4713

dbus-launch --exit-with-session startxfce4 &