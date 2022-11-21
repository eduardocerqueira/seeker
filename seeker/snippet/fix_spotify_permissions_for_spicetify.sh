#date: 2022-11-21T13:55:44Z
#url: https://api.github.com/gists/0dd64586c778f366ac11a72b491bbf43
#owner: https://api.github.com/users/sidevesh

#!/bin/bash
# https://spicetify.app/docs/advanced-usage/installation#spotify-installed-from-flatpak
sudo chmod a+wr /var/lib/flatpak/app/com.spotify.Client/x86_64/stable/active/files/extra/share/spotify
sudo chmod a+wr -R /var/lib/flatpak/app/com.spotify.Client/x86_64/stable/active/files/extra/share/spotify/Apps