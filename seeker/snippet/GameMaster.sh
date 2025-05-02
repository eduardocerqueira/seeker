#date: 2025-05-02T16:32:43Z
#url: https://api.github.com/gists/63231df0791169b7cded3cf44745ac2a
#owner: https://api.github.com/users/Dattebayooooo

#!/bin/bash
# Licensed under the MIT License – https://opensource.org/licenses/MIT
 
DOWNLOAD_LOCATION="/roms"
 
if mountpoint -q /roms2; then
    DOWNLOAD_LOCATION="/roms2"
fi
 
declare -A system_to_url
 
system_to_url["NES"]="https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Nintendo%20Entertainment%20System%20(Headered)/"
system_to_url["SNES"]="https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Super%20Nintendo%20Entertainment%20System/"
system_to_url["N64"]="https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Nintendo%2064%20(BigEndian)/"
 
system_to_url["GB"]="https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Game%20Boy/"
system_to_url["GBC"]="https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Game%20Boy%20Color/"
system_to_url["GBA"]="https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Game%20Boy%20Advance/"
system_to_url["NDS"]="https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Nintendo%20DS%20(Decrypted)/"
 
system_to_url["PSX"]="https://myrient.erista.me/files/Redump/Sony%20-%20PlayStation/"
system_to_url["PSP"]="https://myrient.erista.me/files/Redump/Sony%20-%20PlayStation%20Portable/"
 
system_to_url["GAMEGEAR"]="https://myrient.erista.me/files/No-Intro/Sega%20-%20Game%20Gear/"
system_to_url["GENESIS"]="https://myrient.erista.me/files/No-Intro/Sega%20-%20Mega%20Drive%20-%20Genesis/"
system_to_url["MASTERSYSTEM"]="https://myrient.erista.me/files/No-Intro/Sega%20-%20Master%20System%20-%20Mark%20III/"
system_to_url["SATURN"]="https://myrient.erista.me/files/Redump/Sega%20-%20Saturn/"
system_to_url["DREAMCAST"]="https://myrient.erista.me/files/Redump/Sega%20-%20Dreamcast/"
 
system_to_url["ATC"]=""
system_to_url["AT8"]=""
system_to_url["LYNX"]=""
system_to_url["JAG"]=""
 
system_to_url["NEOGEO"]="https://myrient.erista.me/files/Redump/SNK%20-%20Neo%20Geo%20CD/"
system_to_url["TURBOGRAFX"]="https://myrient.erista.me/files/No-Intro/NEC%20-%20PC%20Engine%20-%20TurboGrafx-16/"
 
SYSTEMS=(
    "NES" "Nintendo Entertainment System"
    "SNES" "Super Nintendo Entertainment System"
    "N64" "Nintendo 64"
 
    "GB" "Game Boy"
    "GBC" "Game Boy Color"
    "GBA" "Game Boy Advance"
    "NDS" "Nintendo DS"
 
    "PSX" "Sony PlayStation"
    "PSP" "PlayStation Portable"
 
    "GAMEGEAR" "Sega Game Gear"
    "GENESIS" "Sega Genesis"
    "MASTERSYSTEM" "Sega Master System"
    "SATURN" "Sega Saturn"
    "DREAMCAST" "Sega Dreamcast"
 
    # "ATC" "Atari 2600"
    # "AT8" "Atari 8-bit"
    # "LYNX" "Atari Lynx"
    # "JAG" "Atari Jaguar"
    
    "NEOGEO" "Neo Geo"
    "TURBOGRAFX" "PC Engine / TurboGrafx-16"
)
 
DIALOG_RC_LOCATION="/tmp/gamemaster-dialog.rc"
 
cat << EOF > "$DIALOG_RC_LOCATION"
use_shadow = OFF
screen_color = (WHITE,BLACK,OFF)
shadow_color = (BLACK,BLACK,OFF)
dialog_color = (YELLOW,BLACK,OFF)
title_color = (RED,BLACK,OFF)
border_color = (CYAN,BLACK,OFF)
border2_color = (CYAN,BLACK,OFF)
button_active_color = (BLACK,WHITE,OFF)
button_inactive_color = (WHITE,BLACK,OFF)
button_key_active_color = (BLACK,WHITE,OFF)
button_key_inactive_color = (WHITE,BLACK,OFF)
button_label_active_color = (BLACK,WHITE,OFF)
button_label_inactive_color = (WHITE,BLACK,OFF)
position_indicator_color = (WHITE,BLACK,OFF)
menubox_color = (WHITE,BLACK,OFF)
menubox_border_color = (BLACK,BLACK,OFF)
menubox_border2_color = (BLACK,BLACK,OFF)
item_color = (WHITE,BLACK,OFF)
item_selected_color = (BLACK,WHITE,OFF)
tag_color = (WHITE,BLACK,OFF)
tag_selected_color = (BLACK,WHITE,OFF)
tag_key_color = (WHITE,BLACK,OFF)
tag_key_selected_color = (BLACK,WHITE,OFF)
EOF
 
setup_dialogs() {
    sudo chmod 666 /dev/tty1
    export DIALOGRC=$DIALOG_RC_LOCATION
 
    if [[ -z $(pgrep -f gptokeyb) ]] && [[ -z $(pgrep -f oga_controls) ]]; then
        sudo chmod 666 /dev/uinput
        export SDL_GAMECONTROLLERCONFIG_FILE="/opt/inttools/gamecontrollerdb.txt"
        /opt/inttools/gptokeyb -1 "dialog" -c "/opt/inttools/keys.gptk" > /dev/null &
        disown
        set_gptokeyb="Y"
    fi
 
    # hide cursor
    printf "\e[?25l" > /dev/tty1
    dialog --clear
 
    height="20"
    width="58"
    items="15"
 
    export TERM=linux
}
 
 
cleanup_dialogs() {
    # Clean up
    if [[ ! -z "$set_gptokeyb" ]]; then
        pgrep -f gptokeyb | sudo xargs kill -9
        unset SDL_GAMECONTROLLERCONFIG_FILE
    fi
 
    export DIALOGRC=
    rm $DIALOG_RC_LOCATION
 
    clear > /dev/tty1
}
 
 
show_system_select() {
    dialog --title "Press Select to Exit" \
        --menu "Choose a System:" $height $width $items "${SYSTEMS[@]}" 2>&1 > /dev/tty1
}
 
show_game_search() {
    osk "Search (Press Select for All Games)"
}
 
show_games() {
    local formatted_games=()
    readarray -t formatted_games <<< "$@"
 
    dialog --no-items --title "Press Select to Exit" \
        --menu "Choose a Game:" $height $width $items "${formatted_games[@]}" 2>&1 > /dev/tty1
 
}
 
# Returns zero if all dependencies are satisfied 
check_dependencies() {
   which fzf > /dev/null && which 7z > /dev/null
}
 
setup_dependencies() {
    sudo apt install fzf p7zip-full -y
}
 
clear
 
# Check internet
if ! sudo ping -q -c 1 -W 1 8.8.8.8 > /dev/null; then
    echo "Error: Not connected to the internet. Exiting......"
    sleep 5
    exit
fi
 
check_dependencies
if [ $? -ne 0 ]; then
    setup_dependencies
fi
 
setup_dialogs

is_game_added=false

while true; do
    system=$(show_system_select)
    clear > /dev/tty1
     
    if [ "$system" = "" ]; then
        break
    fi
     
    url="${system_to_url[$system]}"
    
    gameslist_html="/tmp/$system.html"
    curl_progress=$(mktemp)

    if [ ! -f $gameslist_html ]; then
        curl "$url" > $gameslist_html 2> $curl_progress &
        page_download_pid=$!
    fi

    search=$(show_game_search)
    
    if kill -0 "$page_download_pid" 2> /dev/null; then 
        echo "Please Wait.... Retrieving page" > /dev/tty1

        tail -f $curl_progress &
        tail_pid=$!

        wait $page_download_pid
        kill $tail_pid
    fi

    games=$(grep -oP 'title="\K[^"]+' $gameslist_html)
    
    if [ "$search" ]; then
        games=$(echo "$games" | fzf --filter "$search")
    fi
     
    selected_game=$(show_games "$games")
    clear > /dev/tty1
     
    if [ "$selected_game" = "" ]; then
        continue
    fi
     
    wget "$url$selected_game" -P "$DOWNLOAD_LOCATION/$system" 2>&1 > /dev/tty1
     
    cd "$DOWNLOAD_LOCATION/$system/"
    7z x "$selected_game" 2>&1 > /dev/tty1
    rm "$selected_game"

    is_game_added=true
    break # Remove this line if you want to go to the main menu after the download.
done

cleanup_dialogs

if $is_game_added; then
    sudo systemctl restart emulationstation
fi
