#date: 2022-03-28T17:08:32Z
#url: https://api.github.com/gists/4d2ac94bd770879d8df37c5da0fc7a33
#owner: https://api.github.com/users/armenzg

#!/bin/bash
# This script tries to install PDM on a MacOS host following recommended steps as per https://pdm.fming.dev/#installation
# Currently only supporting Zsh
# This script is designed to allow executing it more than once
set -eu

[[ $(uname) != "Darwin" ]] && echo "Only suppports MacOS setup." && exit 1
[[ "${SHELL}" != "/bin/zsh" ]] && echo "Only suppports zsh shell." && exit 1

# In PDM's output, it is suggested a path to be added to PATH that is very host specific
# which ends up being a symlink, thus, choosing this approach instead
pdm_file_path_basename="${HOME}/Library/Application Support/pdm/venv/bin"

rc_file="${HOME}/.zshrc"
login_file="${HOME}/.zprofile"

promp_user() {
    echo "Continue (y/N)?"
    read -r resp
    case "$resp" in
    y | Y) echo "" ;;
    *)
        echo "Aborted!"
        exit 1
        ;;
    esac
}

# Install PDM if not present
if ! command -v pdm >/dev/null 2>&1; then
    echo "If you care about a customized set up, follow PDM's official docs."
    echo "If you don't care about it, the script will append export command(s) to your shell configuration files."
    promp_user

    # If this file exists, it means that the PATH is not adjusted for this shell
    if ! [ -f "${pdm_file_path_basename}/pdm" ]; then
        curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -
    fi

    # If pdm gets installed and the terminal is not reloaded, the appending of the PATH
    # will not be active, thus, checking before appending since this script may be executed more than once
    if ! grep -qF "pdm/venv/bin" "${rc_file}"; then
        echo "# Added by setup-pdm.sh" >>"${rc_file}"
        # shellcheck disable=SC2016
        echo 'export PATH="${HOME}/Library/Application Support/pdm/venv/bin:$PATH"' >>"${rc_file}"
    fi
    if ! grep -qF "pdm/pep582" "${login_file}"; then
        echo "# Added by setup-pdm.sh" >>"${login_file}"
        pdm --pep582 zsh >>"${login_file}"
    fi
    echo "Run the following command for pdm to become available for this terminal -> exec $SHELL"
fi
