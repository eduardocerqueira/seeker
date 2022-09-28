#date: 2022-09-28T17:18:12Z
#url: https://api.github.com/gists/c2c5e8ec7b3eb58c63e52650908aa67e
#owner: https://api.github.com/users/pecorarista

export FZF_DEFAULT_OPTS="--layout=reverse --info='hidden' --pointer='> ' --no-sort"

function sf() {
    if [ ! -e $HOME/.ssh/config ]
    then
        return
    fi

    local selected=$(
        grep '^Host' $HOME/.ssh/config \
        | sed -e 's/^Host //' \
        | grep -v '^github$' \
        | grep -v '^\*$' \
        | (
            while read -r host
            do
                local user=$(ssh -tt -G $host | grep '^user ' | sed -e 's/^user //')
                local hostname=$(ssh -tt -G $host | grep '^hostname ' | sed -e 's/^hostname //')
                local identityfile=$(
                    ssh -tt -G $host \
                    | grep '^identityfile ' \
                    | sed -e 's/^identityfile //' \
                    | xargs -n1 basename
                )
                echo "💻${host} 🙂${user} 🔑${identityfile} 🌏${hostname}"
            done;
          ) \
        | fzf --prompt 'host> '
    )
    if [ -n "$selected" ]
    then
        print -z "ssh $(echo $selected | sed -e 's/^💻\([^ ]\+\) 🙂\([^ ]\+\) .*/\2@\1/')"
    fi
}

