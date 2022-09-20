#date: 2022-09-20T17:01:58Z
#url: https://api.github.com/gists/fa1896fc845f75f2045d80da15ea6381
#owner: https://api.github.com/users/tsibley

ps1() {
    # Bold, bright white text (fg)…
    printf '\[\e[1;97m\]'

    local -a wordmark=(
        '#4377cd  '
        '#4377cd N'
        '#5097ba e'
        '#63ac9a x'
        '#7cb879 t'
        '#9abe5c s'
        '#b9bc4a t'
        '#d4b13f r'
        '#e49938 a'
        '#e67030 i'
        '#de3c26 n'
        '#de3c26  '
    )

    # …on a colored background
    for tuple in "${wordmark[@]}"; do
        read -r color letter <<<"$tuple"
        bg "$color"
        printf "${letter:- }"
    done

    # Add working dir and traditional prompt char (in magenta)
    printf '\[\e[0m\] \w %s\$ ' "$(fg "#ff00ff")"

    # Reset
    printf '\[\e[0m\]'
}

fg() {
    # ESC[ 38;2;⟨r⟩;⟨g⟩;⟨b⟩ m — Select RGB foreground color
    printf '\[\e[38;2;%d;%d;%dm\]' $(rgb "$1")
}

bg() {
    # ESC[ 48;2;⟨r⟩;⟨g⟩;⟨b⟩ m — Select RGB background color
    printf '\[\e[48;2;%d;%d;%dm\]' $(rgb "$1")
}

rgb() {
    local color="${1###}"
    local r g b

    r=$((0x"${color:0:2}"))
    g=$((0x"${color:2:2}"))
    b=$((0x"${color:4:2}"))

    echo $r $g $b
}
