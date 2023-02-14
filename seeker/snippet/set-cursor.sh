#date: 2023-02-14T17:10:51Z
#url: https://api.github.com/gists/e455a740bea84885130ebe13678dc551
#owner: https://api.github.com/users/nxjosephofficial

#!/usr/bin/env bash

# Cursor in ~/.Xresources
if [ -f "$HOME/.Xresources" ]; then
        cursor="$(cat .Xresources | grep ^Xcursor.theme | awk {'print $2'})"
fi

# Set cursor for GTK
gtk() {
if [ -f "$HOME/.gtkrc-2.0" ]; then
        echo "gtk-cursor-theme-name = ${cursor}" > "$HOME/.gtkrc-2.0"
fi

if [ -f "$HOME/.config/gtk-3.0/settings.ini" ]; then
        cat <<EOF > "$HOME/.config/gtk-3.0/settings.ini"
[Settings]
gtk-cursor-theme-name = "${cursor}"
gtk-cursor-theme-size = 16
EOF
fi
}

xres() {
cat <<EOF > "$HOME/.Xresources"
Xcursor.theme: "$cursor"
EOF
}

# if cursor is not set on ~/.Xresources, ask for cursor name
if [[ "$cursor" == "" && ! -f "$HOME/.Xresources" ]]; then
        read -p "Cursor name: " cursor
        if [ "$cursor" == "" ]; then
                exit
        else
            	xres
                gtk
        fi
else
       	exit
fi