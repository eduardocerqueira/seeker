#date: 2021-11-19T16:55:57Z
#url: https://api.github.com/gists/0aa34d90f3719a8349cc67ab259514e2
#owner: https://api.github.com/users/peytondmurray

git() {
    if [[ "$#" == 1 && "$1" == "log" ]]; then
        local branch="$(command git rev-parse --abbrev-ref HEAD 2>/dev/null)"
        local default_branch="$(get_git_default_branch)"
        if [[ ! -z "${branch}" ]]; then
            if [[ ${branch} == ${default_branch} ]]; then
                git logg
            else
                git logg ${default_branch}..
            fi
        fi
	else
		command git $@
	fi
}

get_git_default_branch() {
    echo $(git branch -rl '*/HEAD' | rev | cut -d/ -f1 | rev)
}
