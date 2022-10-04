#date: 2022-10-04T17:11:18Z
#url: https://api.github.com/gists/6743079deff9f9e5e2fa8973394c8abd
#owner: https://api.github.com/users/jobvite-github

#============================================================
#
#= Usage
#	Parameters
# 		<PID> (Required): The parent task's PID to know when to stop.
# 		<Message> (Optional): The message you want to display. If none, only spinners will show.
# 		<Hex> (Optional): The Hex valued color of the animation.
#
# (
#     (
#         git push origin develop -f 2>/dev/null
#     ) &
#     loadingAnimation $! "Pushing to origin/develop"
# )
#
#============================================================


#============================================================
# tput Text Formatting
#============================================================

#- Colors
Black='#000000'
White='#FFFFFF'
Red='#C8334D'
Orange='#FEA42F'
Yellow='#C7C748'
Green='#43CC63'
Teal='#8AFFC8'
Blue='#88D1FE'
Dark='#615340'

cSuccess=$Green
cWarning=$Orange
cError=$Red
cMessage=$Blue
cEnd=$(tput sgr0)

ALERTWIDTH=30 #Columns for alert width

#- Coloring Functions
function Success() {
    echo -e $(formatText 'bold' "$(colorText $cSuccess $1) ")
}
function Warning() {
    echo -e $(formatText 'bold' "$(colorText $cWarning $1)")
}
function Error() {
    echo -e $(formatText 'bold' "$(colorText $cError $1)")
}
function Message() {
    echo -e $(formatText 'bold' "$(colorText $cMessage $1)")
}
function Color() {
    # $1: text color <required>
    # $2: text content <required>
    echo -e $(formatText 'bold' "$(colorText $1 $2)")
}
function Highlight() {
    # $1: highlight color <required>
    # $2: text content <required>
    # $3: spacing character <optional>
    echo -e $(formatText 'bold' "$(highlightText $1 $2)")
}

#- Formatting Functions
function Alert() {
    txt_color=$Black;
    color=$cWarning;
    space=''
    str=''

    setParams $@

    echo -e $(Highlight $color "$(Color $txt_color $str)")
}
function Prompt() {
    REPLY=''
    color=$Teal;
    str=''

    setParams $@

    Color $color "$str:"
    read
    return $REPLY
}
function setParams() {
    if [[ $3 != "" ]]; then
        txt_color=$1
        color=$2
        str=$3
    elif [[ $2 != "" ]]; then
        color=$1
        str=$2
    else
        str=$1
    fi
}
function setString() {
    space=''
    delimeter=${1:-' '}
    if [[ $delimeter != " " ]]; then
        for (( i = 1; i <= ($ALERTWIDTH-${#str}); i += 1 )); do
            space=$space$delimeter
        done
        str="$str$space"
    else
        for (( i = 1; i <= ($ALERTWIDTH-${#str}) / 2; i += 1 )); do
            space=$space$delimeter
        done
        str="$space$str$space "
    fi
}
function formatText() {
    res=''
    for (( i = 1; i < $#; i += 1 )); do
        if [[ $@[$i] == 'normal' ]]; then
            res=$res"$(tput sgr0)"
        elif [[ $@[$i] == 'bold' ]]; then
            res=$res"$(tput bold)"
        elif [[ $@[$i] == 'underline' ]]; then
            res=$res"$(tput smul)"
        elif [[ $@[$i] == 'nounderline' ]]; then
            res=$res"$(tput rmul)"
        fi
    done

    res=$res$@[$#]

    endFormatting
}
function colorText() {
    # two required arguments, one optional argument for color
    # $1: text color <required>
    # $2: text content <required>

    res=''

    res=$res$(tput setaf $(fromHex $1))
    res=$res${@:2}

    endFormatting
}
function highlightText() {
    # $1: highlight color <required>
    # $2: text content <required>
    color=$1
    str=$2

    setString ${3-'-'}

    res=''

    res=$res$(tput setab $(fromHex $1))
    res="$res $2 "

    endFormatting
}
function fromHex() {
    # fromHex: https://gist.github.com/mhulse/b11e568260fb8c3aa2a8
    hex=$1
    if [[ $hex == "#"* ]]; then
        isHex=true
        hex=$(echo $1 | awk '{print substr($0,2)}')
    fi
    r=$(printf '0x%0.2s' "$hex")
    g=$(printf '0x%0.2s' ${hex#??})
    b=$(printf '0x%0.2s' ${hex#????})
    rgb=$(((r<75?0:(r-35)/40)*6*6+(g<75?0:(g-35)/40)*6+(b<75?0:(b-35)/40)+16))

    echo -e ${rgb:0}
}
function endFormatting() {
    if [[ $(contains $@[$#] $cEnd) ]] || res=$res$cEnd

    echo -e $res
}

# End tput Text Formatting
#============================================================

#============================================================
#
#= Animations
#
#============================================================

function patience() {
    loadingAnimation $! 'Please be patient. Doing a lot of work over here...'
}

#!/bin/bash
# Shows a spinner while another command is running. Randomly picks one of 12 spinner styles.
# @args command to run (with any parameters) while showing a spinner.
#       E.g. ‹spinner sleep 10›

function shutdown() {
    tput cnorm # reset cursor
}

function cursorBack() {
  echo -en "\033[$1D"
}

function loadingAnimation() {
    trap shutdown EXIT

    # make sure we use non-unicode character type locale
    # (that way it works for any locale as long as the font supports the characters)
    local LC_CTYPE=C

    local pid=$1 # Process Id of the previous running command

    case $(($RANDOM % 6)) in
    0)
        local spin='-\|/'
        local charwidth=1
        ;;
    1)
        local spin="▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"
        local charwidth=3
        ;;
    2)
        local spin="▉▊▋▌▍▎▏▎▍▌▋▊▉"
        local charwidth=3
        ;;
    3)
        local spin='▖▘▝▗'
        local charwidth=3
        ;;
    4)
        local spin='◢◣◤◥'
        local charwidth=3
        ;;
    5)
        local spin='◐◓◑◒'
        local charwidth=3
        ;;
    esac

    local i=0
    tput civis # cursor invisible
    while kill -0 $pid 2>/dev/null; do
        local i=$(((i + $charwidth) % ${#spin}))
        if [[ $2 != "" ]]; then
            echo -en "$(tput bold)$(tput setaf $(fromHex ${3=Yellow})) ${spin:$i:$charwidth} $2 ${spin:$i:$charwidth} $(tput sgr0) \r"
        else
            echo -en "$(tput bold)$(tput setaf $(fromHex ${3=Yellow})) ${spin:$i:$charwidth} $(tput sgr0) \r"
        fi

        cursorBack 1
        sleep .1
    done

    wait $pid # capture exit code

    return $?
}

#============================================================