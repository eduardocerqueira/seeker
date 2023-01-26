#date: 2023-01-26T17:08:06Z
#url: https://api.github.com/gists/36e5c97a058e910d7456632208d6db9e
#owner: https://api.github.com/users/dadevel

# append to ~/.zshrc or ~/.bashrc

use-ccache() {
    if [[ "$1" != *.ccache ]]; then
        echo 'bad args' >&2
        return 1
    fi
    export KRB5CCNAME="$(realpath -- "$1")"
}

use-kirbi() {
    case "$1" in
        *.kribi)
            impacket-ticketconvert -- "$1" "${1%.kirbi}.ccache"
            use-ccache "${1%.kirbi}.ccache"
            ;;
        base64:*)
            declare -r tmp="$(mktemp --tmpdir XXXXXXXX.kirbi)"
            base64 -d <<< "$1" > "${tmp}"
            declare -r name="$(impacket-describeticket "${tmp}" | rg '^\[\*\] User Name *: (.*?)$' --replace '$1')"
            if [[ -n "${name}" ]]; then
                mv -i -- "${tmp}" "./${name}.ccache"
                use-ccache "./${name}.ccache"
            else
                rm -f -- "${tmp}"
                echo 'bad args' >&2
            fi
            ;;
        *)
            echo 'bad args' >&2
            return 1
            ;;
    esac
}