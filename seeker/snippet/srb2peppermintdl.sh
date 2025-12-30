#date: 2025-12-30T16:49:16Z
#url: https://api.github.com/gists/2678bf0255755e873ea686cf60abcb00
#owner: https://api.github.com/users/Bijman

#!/bin/sh
set -e

curl -L "https://mb.srb2.org/addons/peppermints-srb2-styled-models.3959/download"-o"/tmp/PEPPERMINTMODELS.zip"
7z x -y "/tmp/PEPPERMINTMODELS.zip" -o"$HOME/.var/app/org.srb2.SRB2/.srb2/models"

if [ -f "$HOME/.var/app/org.srb2.SRB2/.srb2/config.cfg" ]; then

    gawk -i inplace '{gsub("gr_models \"Off\"","gr_models \"On\""); gsub("renderer \"Software\"","renderer \"OpenGL\""); print}' "$HOME/.var/app/org.srb2.SRB2/.srb2/config.cfg"

else

    printf "gr_models \"On\"\nrenderer \"OpenGL\"" > "$HOME/.var/app/org.srb2.SRB2/.srb2/config.cfg"

fi

if [ -f "$HOME/.var/app/org.srb2.SRB2/.srb2/models.dat" ]; then

    printf "\n%s\n" "Found models.dat at path $HOME/.var/app/org.srb2.SRB2/.srb2/models.dat. Do you want to overwrite it with models.dat from model pack? Enter \"yes/Yes\" or \"no/No\" (\"y/Y\" or \"n/N\")."
    read -r CONFIRM

    if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ] || [ "$CONFIRM" = "yes" ] || [ "$CONFIRM" = "Yes" ]; then

        mv -f "$HOME/.var/app/org.srb2.SRB2/.srb2/models/Read Please!.txt" "$HOME/.var/app/org.srb2.SRB2/.srb2/models.dat"

    else

        exit

    fi

fi

printf "\n%s\n" "Done. Model pack is installed. Have fun!"