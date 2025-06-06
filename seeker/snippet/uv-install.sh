#date: 2025-06-06T16:47:52Z
#url: https://api.github.com/gists/fdbfc77b924a08ceab7197d010280dac
#owner: https://api.github.com/users/dotysan

#! /usr/bin/env bash
#
# Simple non-POSIX UV installer.
#

# optional
#DEBUG=yep

main() {
    chk_deps
    get_uv
}

#=======================================================================

chk_deps() {
    chkadep curl
    chkadep jq
    chkadep uname
}

get_uv() {

    # Most modern linux distros will add ~/.local/bin to the $PATH on
    # next login--as soon as it exists. But if we're creating it here,
    # inject the $PATH now, so we don't need to log out & back in!
    if ! grep -qE "(^|:)$HOME/.local/bin:" <<<"$PATH"
    then export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! chkadep uv || ! chkadep uvx
    then
        get_github_latest astral-sh uv
        create_uv_receipt
    fi

    uv self update
}

#-----------------------------------------------------------------------

chkadep() {
    if ! hash "$@"
    then
        echo "ERROR: Must install $@ to use this script."
        return 1
    fi >&2
}

get_github_latest() {
    local owner="$1"
    local repo="$2"

    local os=$(uname --operating-system)
    if ! [[ ${os,,} =~ linux ]]
    then
        echo "ERROR: OS:$os This script only understands linux."
        return 1
    fi >&2

    local latest_url="https://api.github.com/repos/$owner/$repo/releases/latest"
    VERS=$(curl --silent "$latest_url" |jq --raw-output .name)
    local machine=$(uname --machine)
    local asset="https://github.com/$owner/$repo/releases/download/$VERS/$repo-$machine-unknown-linux-gnu.tar.gz"

    mkdir --parents "$HOME/.local/bin"
    curl --silent --location "$asset" |\
        tar --extract --gzip --directory="$HOME/.local/bin" --strip-components=1
}

create_uv_receipt() {
    mkdir --parents "$HOME/.config/uv"
    cat >"$HOME/.config/uv/uv-receipt.json" <<-EOF
	{
	    "binaries": ["uv", "uvx"],
	    "install_layout": "flat",
	    "install_prefix": "$HOME/.local/bin",
	    "modify_path": false,
	    "provider": {
	        "source": "dotysan",
	        "version": "0.1.0"
	    },
	    "source": {
	        "app_name": "uv",
	        "name": "uv",
	        "owner": "astral-sh",
	        "release_type": "github"
	    },
	    "version": "$VERS"
	}
	EOF
}

#=======================================================================

if [ "${BASH_VERSINFO[0]}" != "5" ]
then
    echo 'This script only tested with Bourne-Again Shell v5.'
    exit 1
fi >&2

# poor man's __main__
return 2>/dev/null ||:

set -o errexit
set -o nounset
set -o pipefail

if [[ "${DEBUG:-}" ]]
then
    PS4func() {
        local lineno="$1"
        local i f=''
        local c="\033[0;36m" y="\033[0;33m" n="\033[0m"
        local d=$((${#FUNCNAME[@]}-2))

        if [[ $lineno == 1 ]]
        then lineno=0
        fi

        for ((i=d; i>0; i--))
        do printf -v f "%s%s()" "$f" "${FUNCNAME[i]}"
        done

        local src="${BASH_SOURCE[1]:-}"
        printf "$y%s:%04d$c%s$n " "${src##*/}" "$lineno" "$f"
    }
    PS4='\r$(PS4func $LINENO)'
    set -o xtrace
fi

main
exit 0
