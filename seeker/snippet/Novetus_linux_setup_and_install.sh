#date: 2024-06-03T16:54:58Z
#url: https://api.github.com/gists/2d4b4c0cc550e793ef72cdb27871d7f5
#owner: https://api.github.com/users/KLanausse

#!/bin/bash
prefix=$HOME/.local/share/wineprefixes/Novetus

#Check for an existing install
if [ -d "$prefix" ]; then
    echo "You seem to already have Novetus installed."
    echo "Proceeding will completely wipe your previous install!"
    read -p "Are you sure you want to continue? (y/N) " doReinstall
    case $doReinstall in
        [Yy]* ) echo "Reinstalling...";;
        [Nn]* ) exit 1;;
        * ) exit 1;;
    esac
fi

orrhArchive=$(find . -maxdepth 1 -name 'novetus-windows*.zip' -type f | tail -n 1)

if ((${#orrhArchive} <= 3)); then
    echo "novetus-windows(-beta).zip was not found!"
    echo "Make sure novetus-windows(-beta).zip is in the same folder as the script!"
    exit 1
fi

rm -rf $prefix
rm ~/.local/share/applications/Novetus.desktop
rm ~/Desktop/Novetus.desktop

WINEPREFIX=~/.local/share/wineprefixes/Novetus winetricks -q wininet winhttp mfc80 mfc90 gdiplus wsh56 urlmon pptfonts corefonts dxvk
WINEPREFIX=~/.local/share/wineprefixes/Novetus winetricks wininet=builtin winihttp=native

7z x -t7z $novetushArchive -oNovetus/
mv Novetus/ $prefix/drive_c/ProgramData/

#Cleanup

#Icon
wget https://gist.github.com/assets/66651363/58d0c0b6-18ac-44b8-a390-c6237237975c -O $prefix/drive_c/ProgramData/Novetus/Icon.png

#Create Desktop Shortcut
echo "[Desktop Entry]" >> ~/.local/share/applications/Novetus.desktop
echo "Name=Novetus" >> ~/.local/share/applications/Novetus.desktop
echo "Comment=${novetusArchive:2:-4}" >> ~/.local/share/applications/Novetus.desktop
echo "Icon=$prefix/drive_c/ProgramData/Novetus/Icon.png" >> ~/.local/share/applications/Novetus.desktop
echo "Exec=env WINEPREFIX=\"$HOME/.local/share/wineprefixes/Novetus\" wine C:\\\\\\\\ProgramData\\\\\\\\Novetus\\\\\\\\data\\\\\\\\bin\\\\\\\\Novetus.exe" >> ~/.local/share/applications/Novetus.desktop
echo "Type=Application" >> ~/.local/share/applications/Novetus.desktop
echo "Categories=Games;" >> ~/.local/share/applications/OnlyRetroRobloxHere.desktop
echo "StartupNotify=true" >> ~/.local/share/applications/OnlyRetroRobloxHere.desktop
echo "Path=$HOME/.local/share/wineprefixes/Novetus/drive_c/ProgramData/Novetus" >> ~/.local/share/applications/Novetus.desktop
echo "StartupWMClass=Novetus.exe" >> ~/.local/share/applications/Novetus.desktop
cp ~/.local/share/applications/Novetus.desktop ~/Desktop/Novetus.desktop

#Open
WINEPREFIX=$prefix wine C:\\\\\\\\ProgramData\\\\\\\\OnlyRetroRobloxHere\\\\\\\\OnlyRetroRobloxHere.exe