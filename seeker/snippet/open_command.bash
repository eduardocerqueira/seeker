#date: 2023-11-15T17:08:05Z
#url: https://api.github.com/gists/f9931a66bfc24e40e91aa4c8ec52cdd1
#owner: https://api.github.com/users/MichaelDimmitt

__find_open_command_from_Operating_System() {
    local open_cmd
    OSTYPE="$(uname -s | tr 'A-Z' 'a-z')"

    case "$OSTYPE" in
    darwin*)  open_cmd='open' ;;
    cygwin*)  open_cmd='cygstart' ;;
    linux*)   [[ "$(uname -r)" != *icrosoft* ]] && open_cmd='nohup xdg-open' || {
                open_cmd='cmd.exe /c start ""'
                } ;;
    msys*)    open_cmd='start ""' ;;
    *)        echo "Platform $OSTYPE not supported"
                return 1
                ;;
    esac
    echo "$open_cmd"
}

# run the function and use it to open https://www.google.com, see below:

$(__find_open_command_from_Operating_System) https://www.google.com