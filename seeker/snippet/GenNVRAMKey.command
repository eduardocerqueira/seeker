#date: 2023-11-03T16:55:20Z
#url: https://api.github.com/gists/87a601e0d72e433ab5e0d8b7d1fcdb92
#owner: https://api.github.com/users/ittps-pro

#!/bin/zsh

echo -n "Please specify full path to OpenCore DEBUG logfile (you can drag and drop it in if desired): "
read logfile

if [[ -f "$logfile" ]]; then
    if grep -l -E "macOS Installer.*com.apple.installer" "$logfile" ; then
        echo "Found logfile: $logfile"
        UUID="$(grep -E "macOS Installer.*com.apple.installer" "$logfile" | sed -e 's%^.*/\\\([0-9A-F-]*\)\\com.apple.installer.boot.efi.*%\1%g')"

        echo "macOS Installer disk UUID is: $UUID"

        base64=$(python -c "import base64 ; print(base64.b64encode('msu-product-url://$UUID/macOS%20Install%20Data'))")

        echo "Add the following to your OpenCore config.plist, in section NVRAM -> Add -> 7C436110-AB2A-4BBB-A880-FE41995C9F82"
        echo
        echo "<key>msu-product-url</key>"
        echo "<data>$base64</data>"
        echo
        echo "Then boot OpenCore again, choose the 'macOS Installer' option from the menu, and installation should proceed and complete."
        echo "Remove the 'msu-product-url' NVRAM entry from config.plist once installation is complete."
        else
            echo "ERROR: Could not find macOS Installer entry in provided logfile: $logfile"
        fi
else
    echo "ERROR: Could not find logfile at $logfile"
fi