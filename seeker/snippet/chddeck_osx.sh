#date: 2022-10-14T17:23:31Z
#url: https://api.github.com/gists/4a512bd1d4ae651fd42c17c714617962
#owner: https://api.github.com/users/dillera

#!/usr/local/bin/bash

# chddeck_osx.sh for Macintosh OSX 2022
# original script from https://www.emudeck.com/ EmuDeck installation.
# https://github.com/dragoonDorise/EmuDeck/blob/main/tools/chdconv/chddeck.sh
# modified to run on a local copy of ROM files to compress them before installing on SteamDeck
#
#    N O T E S:
#
# install chdman from brew! You need this for the script to do anything...
# $ brew install rom-tools

# INSTALL BASH 5 from brew! You need this since the bash on OSX is old and no mapfiles...
# $ brew install bash

# Modify the romsPath below to be the roms dir that you have locally...
# I use this rsync command to push file to the SteamDeck:
# $ rsync -ave ssh --exclude '.DS_Store' /Users/myuser/roms deck@192.168.CHANGE.ME:/run/media/mmcblk0p1/Emulation/

# that will push over all the local files you have in the 'roms' dir on your mac to the deck. Ensure that you copy
# the names of the folders on the SteamDeck exactly as they are in the /run/media/mmcblk0p1/Emulation/roms folder.

# Running -
# ensure you run as $ /usr/local/bin/bash ./chddeck_osx.sh so that you get the bash 5 installed from brew.
# but the first line of this file should do that for you as well. 


# CHANGE ME

	romsPath="/Users/CHANGEME/roms"
	toolsPath="/usr/local/bin"
	chdPath="${toolsPath}/"

# NO CHANGES Needed Below

	#initialize log
	TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
	LOGFILE="$chdPath/chdman-$TIMESTAMP.log"
	exec > >(tee "${LOGFILE}") 2>&1


	#ask user if they want to pick manually or run a search for eligible files. Manual will need to ask the user to pick a file, and then it will need to ask the type to convert to. (chd, rvz, cso)


	echo "Checking $romsPath for files eligible for conversion."

	#whitelist
	declare -a chdfolderWhiteList=("dreamcast" "psx" "segacd" "3do" "saturn" "tg-cd" "pcenginecd" "pcfx" "amigacd32" "neogeocd" "megacd" "ps2")
	declare -a rvzfolderWhiteList=("gamecube" "wii")
	declare -a searchFolderList

	export PATH="${chdPath}/:$PATH"

	#find file types we support within whitelist of folders
	for romfolder in "${chdfolderWhiteList[@]}"; do
		echo "Checking ${romsPath}/${romfolder}/"
		mapfile -t files < <(find "${romsPath}/${romfolder}/" -type f -iname "*.gdi" -o -type f -iname "*.cue" -o -type f -iname "*.iso")
		if [ ${#files[@]} -gt 0 ]; then
			echo "found in $romfolder"
			romfolders+=("$romfolder")
		fi
	done

	if (( ${#romfolders[@]} == 0 )); then
		echo "No eligible files found."
	fi


	#CHD
	for romfolder in "${romfolders[@]}"; do
        if [[ " ${chdfolderWhiteList[*]} " =~ " ${romfolder} " ]]; then

            find "$romsPath/$romfolder" -type f -iname "*.gdi" | while read -r f
                do
                    echo "Converting: $f"
                    CUEDIR="$(dirname "${f}")"
                    chdman createcd -i "$f" -o "${f%.*}.chd" && successful="true"
                    if [[ $successful == "true" ]]; then
                        echo "successfully created ${f%.*}.chd"
                        find "${CUEDIR}" -maxdepth 1 -type f | while read -r b
                            do
                                fileName="$(basename "${b}")"
                                found=$(grep "${fileName}" "${f}")
                                if [[ ! $found = '' ]]; then
                                    echo "Deleting ${b}"
                                    rm "${b}"
                                fi
                            done
                            rm "${f}"
                    else
                        echo "Conversion of ${f} failed."
                    fi

                done
            find "$romsPath/$romfolder" -type f -iname "*.cue" | while read -r f
                do
                    echo "Converting: $f"
                    CUEDIR="$(dirname "${f}")"
                    chdman createcd -i "$f" -o "${f%.*}.chd" && successful="true"
                    if [[ $successful == "true" ]]; then
                        echo "successfully created ${f%.*}.chd"
                        find "${CUEDIR}" -maxdepth 1 -type f | while read -r b
                            do
                                fileName="$(basename "${b}")"
                                found=$(grep "${fileName}" "${f}")
                                if [[ ! $found = '' ]]; then
                                    echo "Deleting ${b}"
                                    rm "${b}"
                                fi
                            done
                            rm "${f}"
                    else
                        echo "Conversion of ${f} failed."
                    fi

                done
            find "$romsPath/$romfolder" -type f -iname "*.iso" | while read -r f; do echo "Converting: $f"; chdman createcd -i "$f" -o "${f%.*}.chd" && rm -rf "$f"; done;
        fi
	done

	#rvz

    for romfolder in "${romfolders[@]}"; do
        if [[ " ${rvzfolderWhiteList[*]} " =~ " ${romfolder} " ]]; then
            find "$romsPath/$romfolder" -type f -iname "*.gcm"  -o -type f -iname "*.iso" | while read -r f; do echo "Converting: $f"; /var/lib/flatpak/app/org.DolphinEmu.dolphin-emu/current/active/files/bin/dolphin-tool convert -f rvz -b 131072 -c zstd -l 5 -i "$f" -o "${f%.*}.rvz"  && rm -rf "$f"; done;
        fi
    done



echo -e "Exit" &>> /dev/null


