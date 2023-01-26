#date: 2023-01-26T16:58:30Z
#url: https://api.github.com/gists/0f9c489eedfd1cb6189c3f4912551378
#owner: https://api.github.com/users/krscott

if [[ -z "$PROMPT_HOSTNAME" ]]; then
    PROMPT_HOSTNAME="\h"
fi

# Set if using __git_ps1
# GIT_PS1_SHOWDIRTYSTATE=1
# GIT_PS1_SHOWSTASHSTATE=1
# GIT_PS1_SHOWUNTRACKEDFILES=1

[ -s "/etc/bash_completion.d/git" ] && source /etc/bash_completion.d/git


### Functions

__jobs() {
    local n=$(jobs -p | wc -l | tr -d ' ')
    if [[ $n != 0 ]]; then
        printf "($n) "
    fi
}

# __retval() {
#     local retval=$?
#     [[ $retval != 0 ]] && printf "$retval "
# }

# __git_dirty_modified() {
#     git diff --no-ext-diff --quiet 2>/dev/null || printf "*"
# }

# __git_dirty_staged() {
#     git diff --no-ext-diff --cached --quiet 2>/dev/null || printf "+"
# }

# __git_branch() {
#     git rev-parse --quiet --verify --abbrev-ref HEAD 2>/dev/null
# }

__git_info() {
    local branch=$(git rev-parse --quiet --verify --abbrev-ref HEAD 2>/dev/null)

    if [[ -n $branch ]]; then
        printf -- $branch
        git diff-index --no-ext-diff --quiet HEAD 2>/dev/null || printf '*'
        git diff-index --no-ext-diff --cached --quiet HEAD 2>/dev/null || printf '+'
    fi
}

__git_info_lite() {
    local branch=$(git rev-parse --quiet --verify --abbrev-ref HEAD 2>/dev/null)

    if [[ -n $branch ]]; then
        printf "($branch)"
    fi
}

# build_bash_ps1 is used as a closure for local variables, and is only called once
build_bash_ps1() {
    isdef() {
        type "$1" > /dev/null 2>&1
    }

    local NONE="\[\033[0m\]"    # unsets color to term's fg color

    local CL="\[\033[0;3" # Normal color
    local EM="\[\033[1;3" # Emphasized color
    local BG="\[\033[1;4" # Background color

    local K="0m\]"  # black
    local R="1m\]"  # red
    local G="2m\]"  # green
    local Y="3m\]"  # yellow
    local B="4m\]"  # blue
    local M="5m\]"  # magenta
    local C="6m\]"  # cyan
    local W="7m\]"  # white

    # local UC=$W                 # user's color
    # [ $UID -eq "0" ] && UC=$R   # root's color

    # Prompt color
    local PCL="$R"

    # User string
    # local USER="$CL$PCL\u$CL$PCL@"
    local USER=""

    # Directory string
    #local DIR="\W"  # Current directory name only
    local DIR="\w"  # Full path

    # Space before '$' (usually a space or newline)
    #local PROMPT_WHITESPACE=" "
    # The string "\n" does not work in latest Windows/Git Bash (WTF?)
    # Use literal $'\n' instead
    local PROMPT_WHITESPACE=$'\n'

    case "$HOSTNAME" in
    OLA-2WDFTN2)
        LIGHT_PROMPT=1
        ;;
    esac

    case "$(id -u -n)" in
    kris|Kris|kscot)
        local PCL="$Y"
        ;;
    kscott)
        local PCL="$M"
        ;;
    pi)
        local PCL="$C"
        ;;
    root)
        local USER="$CL$W$BG$R\u$NONE@$CL$PCL"
        ;;
    *)
        local USER="\u@"
        ;;
    esac

    # if isdef jobs; then
    #     local JOBS="$NONE\$(__jobs)"
    # fi
    local JOBS=""

    # local RETVAL="$EM$R\$(__retval)"
    local RETVAL="$EM$R\$([[ \$? == 0 ]] || printf \"\$? \")"

    if isdef git; then
        # local GIT_PS1="$EM$W\$(__git_ps1 \" %s\")"
        local GIT_PS1=" $EM$W\$(__git_info)"
    fi

    # Remove features from slow machines
    if [ -n "$LIGHT_PROMPT" ]; then
        local JOBS=""
        local GIT_PS1=" $EM$W\$(__git_info_lite)"
    fi

    PS1="$RETVAL$JOBS$CL$PCL$USER$PROMPT_HOSTNAME$CL$W:$EM$PCL$DIR$GIT_PS1$NONE$PROMPT_WHITESPACE$ "
}

build_bash_ps1
unset build_bash_ps1
