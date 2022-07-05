#date: 2022-07-05T16:45:56Z
#url: https://api.github.com/gists/4dfe24590a56434139744fb7d1bc6ce9
#owner: https://api.github.com/users/usrbinkat

talosctl reset \
  --talosconfig ./talos/talosconfig \
  --system-labels-to-wipe=EPHEMERAL \
  --system-labels-to-wipe=STATE \
  --reboot --graceful=false \
  --nodes 192.168.1.71 -e 192.168.1.71
talosctl reset \
  --talosconfig ./talos/talosconfig \
  --system-labels-to-wipe=EPHEMERAL \
  --system-labels-to-wipe=STATE \
  --reboot --graceful=false \
  --nodes 192.168.1.72 -e 192.168.1.72
talosctl reset \
  --talosconfig ./talos/talosconfig \
  --system-labels-to-wipe=EPHEMERAL \
  --system-labels-to-wipe=STATE \
  --reboot --graceful=false \
  --nodes 192.168.1.73 -e 192.168.1.73
talosctl reset \
  --talosconfig ./talos/talosconfig \
  --system-labels-to-wipe=EPHEMERAL \
  --system-labels-to-wipe=STATE \
  --reboot --graceful=false \
  --nodes 192.168.1.74 -e 192.168.1.74
talosctl reset \
  --talosconfig ./talos/talosconfig \
  --system-labels-to-wipe=EPHEMERAL \
  --system-labels-to-wipe=STATE \
  --reboot --graceful=false \
  --nodes 192.168.1.75 -e 192.168.1.75
talosctl reset \
  --talosconfig ./talos/talosconfig \
  --system-labels-to-wipe=EPHEMERAL \
  --system-labels-to-wipe=STATE \
  --reboot --graceful=false \
  --nodes 192.168.1.76 -e 192.168.1.76