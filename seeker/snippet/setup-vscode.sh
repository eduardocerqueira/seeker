#date: 2023-10-13T17:06:46Z
#url: https://api.github.com/gists/34c9a25a7ff85baa0df9db335573d01e
#owner: https://api.github.com/users/Knuds1

cat << EOF | sudo tee /usr/local/bin/xdg-open
#!/bin/sh
exec flatpak-spawn --host -- xdg-open \$@
EOF
sudo chmod +x /usr/local/bin/xdg-open

cp /usr/share/applications/code.desktop /usr/share/applications/code-url-handler.desktop ~/.local/share/applications/
TOOLBOX_NAME=$(cat /run/.containerenv | grep 'name=' | sed -e 's/^name="\(.*\)"$/\1/')
sed -i "s/Exec=\/usr\/share\/code\/code/Exec=\/usr\/bin\/toolbox run -c \"$TOOLBOX_NAME\" code/g" ~/.local/share/applications/code.desktop ~/.local/share/applications/code-url-handler.desktop