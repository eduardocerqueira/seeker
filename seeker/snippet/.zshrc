#date: 2022-06-30T17:10:14Z
#url: https://api.github.com/gists/93aad0e92d2d6086d70f8996cc020012
#owner: https://api.github.com/users/tapickell

# Enable Powerlevel10k instant prompt. Should stay close to the top of ~/.zshrc.
# Initialization code that may require console input (password prompts, [y/n]
# confirmations, etc.) must go above this block; everything else may go below.
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# If you come from bash you might have to change your $PATH.
# export PATH=$HOME/bin:/usr/local/bin:$PATH

# Path to your oh-my-zsh installation.
export ZSH="/home/todd.pickell/.oh-my-zsh"

# Set name of the theme to load --- if set to "random", it will
# load a random theme each time oh-my-zsh is loaded, in which case,
# to know which specific one was loaded, run: echo $RANDOM_THEME
# See https://github.com/ohmyzsh/ohmyzsh/wiki/Themes
#ZSH_THEME="muse"

ZSH_THEME="powerlevel10k/powerlevel10k"
# Set list of themes to pick from when loading at random
# Setting this variable when ZSH_THEME=random will cause zsh to load
# a theme from this variable instead of looking in $ZSH/themes/
# If set to an empty array, this variable will have no effect.
# ZSH_THEME_RANDOM_CANDIDATES=( "robbyrussell" "agnoster" )

# Uncomment the following line to use case-sensitive completion.
# CASE_SENSITIVE="true"

# Uncomment the following line to use hyphen-insensitive completion.
# Case-sensitive completion must be off. _ and - will be interchangeable.
# HYPHEN_INSENSITIVE="true"

# Uncomment the following line to disable bi-weekly auto-update checks.
# DISABLE_AUTO_UPDATE="true"

# Uncomment the following line to automatically update without prompting.
# DISABLE_UPDATE_PROMPT="true"

# Uncomment the following line to change how often to auto-update (in days).
# export UPDATE_ZSH_DAYS=13

# Uncomment the following line if pasting URLs and other text is messed up.
# DISABLE_MAGIC_FUNCTIONS="true"

# Uncomment the following line to disable colors in ls.
# DISABLE_LS_COLORS="true"

# Uncomment the following line to disable auto-setting terminal title.
# DISABLE_AUTO_TITLE="true"

# Uncomment the following line to enable command auto-correction.
# ENABLE_CORRECTION="true"

# Uncomment the following line to display red dots whilst waiting for completion.
# Caution: this setting can cause issues with multiline prompts (zsh 5.7.1 and newer seem to work)
# See https://github.com/ohmyzsh/ohmyzsh/issues/5765
# COMPLETION_WAITING_DOTS="true"

# Uncomment the following line if you want to disable marking untracked files
# under VCS as dirty. This makes repository status check for large repositories
# much, much faster.
# DISABLE_UNTRACKED_FILES_DIRTY="true"

# Uncomment the following line if you want to change the command execution time
# stamp shown in the history command output.
# You can set one of the optional three formats:
# "mm/dd/yyyy"|"dd.mm.yyyy"|"yyyy-mm-dd"
# or set a custom format using the strftime function format specifications,
# see 'man strftime' for details.
# HIST_STAMPS="mm/dd/yyyy"

# Would you like to use another custom folder than $ZSH/custom?
# ZSH_CUSTOM=/path/to/new-custom-folder

# Which plugins would you like to load?
# Standard plugins can be found in $ZSH/plugins/
# Custom plugins may be added to $ZSH_CUSTOM/plugins/
# Example format: plugins=(rails git textmate ruby lighthouse)
# Add wisely, as too many plugins slow down shell startup.
plugins=(
  asdf
  aws
  brew
  colored-man-pages
  command-not-found
  docker
  git
  git-extras
  git-prompt
  history
  iterm2
  jsontools
  last-working-dir
  linux
  mix
  mosh
  node
  npm
  nvm
  pip
  postgres
  rebar
  redis-cli
  sudo
  themes
  wd
  web-search
)

# my alias
alias list="ls -hAlF"
alias clip="xclip -selection clipboard"
alias unclip="xclip -o"
alias batt="bat --theme ansi-light"
alias manc="man 3"
alias gocode="cd ~/code"
alias gita="git add ."
alias gitr="git rm"
alias gitc="git commit -m "
alias gitlol='git commit -am "$(curl -s whatthecommit.com/index.txt)"'
alias gitlog='git log --decorate --graph --pretty=format:"%C(auto)%h%dCreset %C(cyan)(%cr)%Creset %s"'
alias gits="git status"
alias grep='GREP_COLOR="1;37;41" LANG=C grep --color=auto'
alias ttop="top -ocpu -R -F -s 2 -n30"
alias gotmux="tmux new -s"
alias ls='ls -G'
alias ll='ls -la'
alias la='ls -a'
alias grep='GREP_COLOR="1;37;41" LANG=C grep --color=auto'
alias pgrep='ps aux | grep'
alias tn="tmux -2 new-session -s"
alias tt="tmux -2 attach -t"
alias myip="curl -s ifconfig.me"
alias myspeed="curl -o /dev/null http://speedtest.wdc01.softlayer.com/downloads/test10.zip"
alias whosonport="sudo lsof -i -P | grep"
alias find_grep="find . -type f -print0 | xargs -0 grep -l"
alias rezsh='source ~/.zshrc'
alias hist='history -E'
alias hgrep='history -E | grep'
alias nv='nvim'

setopt share_history

function my_funcs {
  cat ~/.zshrc | grep -v 'grep ' | grep 'usage: ' | sed 's/";//' | awk '{$1="";$2=$3;$3=" :: "; print $0 }'
}

function gitsdir {
  ls | grep "$1" | while read -r line ; do cd $line; echo $line; gits; git log -1 --pretty=format:"Last Commit%n%h%n%an%n%ae%n%ar%n%s%n%b%n" |& cat; echo "\n"; gocode; done
}

# source $ZSH/oh-my-zsh.sh
#source /usr/local/opt/zsh-vi-mode/share/zsh-vi-mode/zsh-vi-mode.plugin.zsh

# User configuration

# export MANPATH="/usr/local/man:$MANPATH"

# You may need to manually set your language environment
export LANG=en_US.UTF-8
# You don't strictly need this collation, but most technical people
# probably want C collation for sane results
export LC_COLLATE=C


# Preferred editor for local and remote sessions
if [[ -n $SSH_CONNECTION ]]; then
  export EDITOR='vim'
else
  export EDITOR='nvim'
fi

# Compilation flags
# export ARCHFLAGS="-arch x86_64"

# Set personal aliases, overriding those provided by oh-my-zsh libs,
# plugins, and themes. Aliases can be placed here, though oh-my-zsh
# users are encouraged to define aliases within the ZSH_CUSTOM folder.
# For a full list of active aliases, run `alias`.
#

export KERL_CONFIGURE_OPTIONS="--without-javac --with-ssl=/opt/homebrew/Cellar/openssl@1.1/1.1.1l"
export KERL_BUILD_DOCS=yes
export ERL_AFLAGS="-kernel shell_history enabled"

# Example aliases
alias zshconfig="nvim ~/.zshrc"
alias nvmconfig="nvim ~/.config/nvim/init.vim"

# export AWS_ACCESS_KEY_ID="AKIARMAOPY62XBC5GXMS"
# export AWS_SECRET_ACCESS_KEY="pvrn62DN6PD2r5BKaXJEgMYqUk2T85y+r54WeU9p"
# export AWS_DEFAULT_REGION="us-east-1"

# . /usr/local/opt/asdf/asdf.sh

source $HOME/.cargo/env

eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
source /home/linuxbrew/.linuxbrew/opt/powerlevel10k/powerlevel10k.zsh-theme

# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh.
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh

. /home/linuxbrew/.linuxbrew/opt/asdf/libexec/asdf.sh
