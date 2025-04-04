#date: 2025-04-04T16:47:35Z
#url: https://api.github.com/gists/4d8ba7c5d08adefd9040a0b1a7d14fa6
#owner: https://api.github.com/users/tkim90

# cd without writing 'cd'
setopt AUTO_CD

plugins=(zsh-autosuggestions)
ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE='fg=8'

#------------------------------------------------------
# Tae's Config
#------------------------------------------------------
alias c="open $1 -a \"Cursor\""

alias spark='cd ~/Documents/spark'
alias be='cd ~/Documents/projects/spark-backend'
alias fe='cd ~/Documents/projects/spark-frontend'
alias browser='cd ~/Documents/projects/spark-browser-agent'

# include hidden files, colorized, human-readable filesize
alias gd='git diff'
alias projects='cd ~/Documents/projects'
alias bsl='brew services list'
alias gs='git status'
alias gco='git checkout'
alias gb='git branch'
alias gp='git pull'
alias gf='git fetch --all'
alias gfa='git fetch --all'
alias gco='git checkout'
alias l='tree -a -L 1 -C -F -h -F'
alias l2='tree -a -L 2 -C -F'
alias editzsh='vi ~/.zshrc'
alias resetzsh='source ~/.zshrc'
alias cat=bat

# Open Cursor from terminal
function cursor {
  open -a "/Applications/Cursor.app" "$@"
}

source $(brew --prefix)/share/zsh-autosuggestions/zsh-autosuggestions.zsh

# Make it so fzf by default also searches in hidden . files like .env
export FZF_DEFAULT_COMMAND='find .'

if [[ $FIND_IT_FASTER_ACTIVE -eq 1 ]]; then
  FZF_DEFAULT_COMMAND='find .'
fi



# alt left + right to skip words
# Disable Alt+arrow for pane switching in tmux
bindkey "^[[1;3D" backward-word    # Alt+left
bindkey "^[[1;3C" forward-word     # Alt+right

# if you want to profile your zsh startup time
# uncomment the following line and run zprof as the first command in a new shell
# zmodload zsh/zprof




# use 256 color terminal
export TERM=xterm-256color

# use vim as standard editor
export VISUAL=nvim
export EDITOR="$VISUAL"

# I'm a weirdo. I like vim but prefer emacs mode on the terminal.
# Since zsh automatically enables vi mode when you set 'vi' as your standard $EDITOR, let's explicitly request emacs mode.
bindkey -e


#------------------------------------------------------
# Better History
#------------------------------------------------------
setopt SHARE_HISTORY        # share history between all sessions
setopt HIST_IGNORE_SPACE    # don't record commands that start with a space
setopt INC_APPEND_HISTORY   # write to $HISTFILE immediately, not just when exiting the shell
setopt HIST_IGNORE_ALL_DUPS # remove old duplicates from history
setopt HIST_VERIFY          # don't execute immediately when picking from history
HISTSIZE=50000              # store more than the default 10_000 entries
SAVEHIST=$HISTSIZE          # and also store all these entries in our $HISTFILE

#------------------------------------------------------
# Aliases
#------------------------------------------------------
alias serve='python -m http.server'
alias de='setxkbmap de'
alias us='setxkbmap us'
alias lnks='~/.bookmarks/lnks.sh'

alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../../'

alias ls="ls --color=auto"
alias ll="ls -asl"

# print current week number
alias week='date +%V'

# use nvim if available
if [ -x "$(command -v nvim)" ]; then
    alias vim='nvim'
fi

#------------------------------------------------------
# Functions
#------------------------------------------------------

# Find out what's running on a given port
whatsonport() {
    lsof -i tcp:$1
}

# load OS specific config
case `uname` in
  Darwin)
    source $HOME/.zshrc-mac
  ;;
  Linux)
    source $HOME/.zshrc-linux
  ;;
  FreeBSD)
    # commands for FreeBSD go here
  ;;
esac


export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

#------------------------------------------------------
# Autocompletion
#------------------------------------------------------

zmodload zsh/complist
autoload -U compinit; compinit
_comp_options+=(globdots)   # include hidden files
setopt MENU_COMPLETE        # Automatically highlight first element of completion menu
setopt AUTO_LIST            # Automatically list choices on ambiguous completion.


# Use select menu for completions
zstyle ':completion:*' menu select

# Autocomplete options when completing a '-'
zstyle ':completion:*' complete-options true

# Style group names a little nicer
zstyle ':completion:*:*:*:*:descriptions' format '%F{green}↓ %d %f'

# Group completion results by type
zstyle ':completion:*' group-name ''

# Set up fzf for general auto-completion shenanigans, if it's installed
FZF_CONFIG=~/.fzf.sh
if [[ -x "$(command -v fzf)" ]] && [[ -f "$FZF_CONFIG" ]]; then
  source "$FZF_CONFIG"
fi

#------------------------------------------------------
# Additional tools (version managers, CLI tools, ...)
#------------------------------------------------------

# Prompt
eval "$(starship init zsh)"

# direnv
eval "$(direnv hook zsh)"

# Keeping old as backup
#export PATH=/opt/homebrew/bin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin:/Applications/iTerm.app/Contents/Resources/utilities:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/Users/taekim/Library/Python/3.9/bin/

# Define core system paths
export PATH="\
/opt/homebrew/bin:\
/usr/local/bin:\
/usr/local/sbin:\
/usr/bin:\
/usr/sbin:\
/bin:\
/sbin:\
/System/Cryptexes/App/usr/bin:\
/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:\
/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:\
/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin"

# Add user-specific paths
export PATH="\
$HOME/.rbenv/shims:\
$HOME/.cargo/bin:\
$HOME/Library/Python/3.9/bin:\
/Applications/iTerm.app/Contents/Resources/utilities:\
$PATH"

# For Ruby
eval "$(rbenv init -)"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh

# Added by Windsurf
export PATH="/Users/taekim/.codeium/windsurf/bin:$PATH"
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"


# pnpm
export PNPM_HOME="/Users/taekim/Library/pnpm"
case ":$PATH:" in
  *":$PNPM_HOME:"*) ;;
  *) export PATH="$PNPM_HOME:$PATH" ;;
esac
# pnpm end
export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"
