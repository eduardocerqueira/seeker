#date: 2022-07-25T16:52:43Z
#url: https://api.github.com/gists/530d569caea666d5dfba572b8676f053
#owner: https://api.github.com/users/pojntfx

sudo usermod -aG wireshark $USER
sudo loginctl terminate-user $USER # Will log you out; logging yourself out with GDM does not work!