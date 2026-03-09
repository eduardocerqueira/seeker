#date: 2026-03-09T17:29:08Z
#url: https://api.github.com/gists/862d9dd0f9bbbcef0921d8de0f8b3c5f
#owner: https://api.github.com/users/kungfusheep

# fast-prompt: zero-fork zsh prompt with git branch + worktree support
#
# shows: repo/subdir:branch (git) or full path (non-git)
# style: dim text, block cursor prompt
#
# usage: source this file in .zshrc

setopt PROMPT_SUBST

_prompt_update() {
    local d=$PWD head root line
    while [[ -n "$d" ]]; do
        if [[ -f "$d/.git/HEAD" ]]; then
            read -r head < "$d/.git/HEAD"
            root="${d:t}"
            if [[ "$PWD" == "$d" ]]; then
                PROMPT_INFO="${root}:${head#ref: refs/heads/}"
            else
                PROMPT_INFO="${root}/${PWD#$d/}:${head#ref: refs/heads/}"
            fi
            return
        elif [[ -f "$d/.git" ]]; then
            read -r line < "$d/.git"
            line="${line#gitdir: }"
            if [[ -f "$line/HEAD" ]]; then
                read -r head < "$line/HEAD"
                root="${d:t}"
                if [[ "$PWD" == "$d" ]]; then
                    PROMPT_INFO="${root}:${head#ref: refs/heads/}"
                else
                    PROMPT_INFO="${root}/${PWD#$d/}:${head#ref: refs/heads/}"
                fi
                return
            fi
        fi
        d=${d%/*}
    done
    PROMPT_INFO="$PWD"
}

precmd_functions+=(_prompt_update)
PS1=$'\n%{\e[2m%}${PROMPT_INFO}\n■ %{\e[0m%}'

