#date: 2021-10-07T16:52:47Z
#url: https://api.github.com/gists/1db5a92174aae00b12a54420dbb050f3
#owner: https://api.github.com/users/L3afMe

# Setup directories
__BASE_DIR="$HOME/Public"

## Config directory
export XDG_CONFIG_HOME="$__BASE_DIR/Config"

## Misc directories
__MISC_BASE_DIR="$__BASE_DIR/Etc"
export POWERCORD_DIR="$__BASE_DIR/Programs/powercord"
export GOPATH="$__MISC_BASE_DIR/Go"
export PYTHONUSERBASE="$__MISC_BASE_DIR/Python"

## Xorg directories
__XORG_BASE_DIR="$__BASE_DIR/Xorg"

### Xorg User directories
export XDG_DESKTOP_DIR="$__XORG_BASE_DIR/Documents"
export XDG_DOCUMENTS_DIR="$__XORG_BASE_DIR/Documents"
export XDG_DOWNLOAD_DIR="$__XORG_BASE_DIR/Downloads"

### Xorg Media directories
__XORG_MEDIA_BASE_DIR="$__BASE_DIR/Xorg"
export XDG_MUSIC_DIR="$__XORG_MEDIA_BASE_DIR/Music"
export XDG_PICTURES_DIR="$__XORG_MEDIA_BASE_DIR/Pictures"
export XDG_VIDEOS_DIR="$__XORG_MEDIA_BASE_DIR/Videos"

### Xorg Hidden directories
__XORG_HIDDEN_BASE_DIR="$__XORG_BASE_DIR/Hidden"
export XDG_CACHE_HOME="$__XORG_HIDDEN_BASE_DIR/Cache"
export XDG_DATA_HOME="$__XORG_HIDDEN_BASE_DIR/Data"
export XDG_STATE_HOME="$__XORG_HIDDEN_BASE_DIR/State"

PATH="$PATH:"\
"/usr/lib/jvm/java-11-graalvm/bin"

# Store ZDOTDIR
ZD=${ZDOTDIR:-$HOME}

# cd without cd
setopt autocd

# Unset some shit
unsetopt beep extendedglob nomatch notify

# Vim binds
bindkey -v

# Set prompt
PROMPT=" %2~ > "

# Setup plugin manager
ZINITDIR="$ZD/.zinit"
if [[ ! -d $ZINITDIR ]]; then
  echo " -- Installing zinit plugin manager --"
  mkdir $ZINITDIR
  git clone https://github.com/zdharma/zinit.git $ZINITDIR/bin
fi
. $ZINITDIR/bin/zinit.zsh

# Lazy load plugins
zinit wait lucid for \
 atinit"ZINIT[COMPINIT_OPTS]=-C; zicompinit; zicdreplay" \
    zdharma/fast-syntax-highlighting \
 blockf \
    zsh-users/zsh-completions \
 atload"!_zsh_autosuggest_start" \
    zsh-users/zsh-autosuggestions

# Ngl can't remember what this does
zstyle :compinstall filename '/home/l3af/.zshrc'

# Load completions
autoload -Uz compinit
compinit

# Load more plugns (requried after compinit)
zinit load hlissner/zsh-autopair

# Automatically run ls after cding if less than 20 files
function cd() {
  builtin cd $@ && if (( $(ls | wc -l) < 20 )); then ls; fi
}

function paste() {
  local file=${1:-/dev/stdin}
  curl --data-binary @${file} https://paste.rs
}