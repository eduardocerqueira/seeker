#date: 2023-10-25T16:51:18Z
#url: https://api.github.com/gists/ae1505da1e38e2d8150dd5dc290df383
#owner: https://api.github.com/users/TheFenrisLycaon

if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

export ZSH="$HOME/.oh-my-zsh"

ZSH_THEME="powerlevel10k/powerlevel10k"

zstyle ':omz:update' mode auto      # update automatically without asking

# Uncomment the following line to change how often to auto-update (in days).
zstyle ':omz:update' frequency 30

COMPLETION_WAITING_DOTS="true"

plugins=(git zsh-syntax-highlighting zsh-autosuggestions zsh-completions)

source $ZSH/oh-my-zsh.sh

export LANG=en_US.UTF-8

# Preferred editor for local and remote sessions
if [[ -n $SSH_CONNECTION ]]; then
  export EDITOR='vim'
else
  export EDITOR='vim'
fi

# Compilation flags
export ARCHFLAGS="-arch x86_64"

alias zshconfig="vim ~/.zshrc"
alias ohmyzsh="vim ~/.oh-my-zsh"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/fenris/.condahome/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/fenris/.condahome/etc/profile.d/conda.sh" ]; then
        . "/home/fenris/.condahome/etc/profile.d/conda.sh"
    else
        export PATH="/home/fenris/.condahome/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

### ARCHIVE EXTRACTION
# usage: ex <file>
ex ()
{
  if [ -f $1 ] ; then
    case $1 in
      *.tar.bz2)   tar xjf $1   ;;
      *.tar.gz)    tar xzf $1   ;;
      *.bz2)       bunzip2 $1   ;;
      *.rar)       unrar x $1   ;;
      *.gz)        gunzip $1    ;;
      *.tar)       tar xf $1    ;;
      *.tbz2)      tar xjf $1   ;;
      *.tgz)       tar xzf $1   ;;
      *.zip)       unzip $1     ;;
      *.Z)         uncompress $1;;
      *.7z)        7z x $1      ;;
      *.deb)       ar x $1      ;;
      *.tar.xz)    tar xf $1    ;;
      *.tar.zst)   unzstd $1    ;;
      *)           echo "'$1' cannot be extracted via ex()" ;;
    esac
  else
    echo "'$1' is not a valid file"
  fi
}

# ls
alias ls='exa -al --color=always --group-directories-first' # my preferred listing
alias la='exa -a --color=always --group-directories-first'  # all files and dirs
alias ll='exa -l --color=always --group-directories-first'  # long format
alias lt='exa -aT --color=always --group-directories-first' # tree listing
alias l.='exa -a | egrep "^\."'				    # dotfiles only

# frequents
alias q='exit'
alias c='clear'
alias h='history'
alias cat='bat'

# git
alias gcl='git clone '
alias gp='git pull'
alias ga='git add '
alias gc='git commit -m '
alias gd='git push'

# Dotfiles Git
alias dg='/usr/bin/git --git-dir=/home/fenris/.cfg/ --work-tree=/home/fenris'
alias dga='/usr/bin/git --git-dir=/home/fenris/.cfg/ --work-tree=/home/fenris add'
alias dgc='/usr/bin/git --git-dir=/home/fenris/.cfg/ --work-tree=/home/fenris commit -m '
alias dgp='/usr/bin/git --git-dir=/home/fenris/.cfg/ --work-tree=/home/fenris push'
alias dgupdate='~/Apps/dotfiles/autoUpdate.sh'

# python
alias pp='python3'
alias insp='pip3 install '

# easy edits
alias vim='vim'
alias vimrc='vim ~/.vimrc'
alias bashrc='vim ~/.bashrc'

# pacman shortcuts
alias mirrorupdate='curl -s "https://archlinux.org/mirrorlist/?country=US&country=GB&protocol=https&use_mirror_status=on" | sed -e 's/^#Server/Server/' -e '/^#/d' | rankmirrors -n 10 -' # Automatically updates mirrors.
alias ins='sudo pacman -S --needed' 
alias insy='yay -S '
alias unins='sudo pacman -Rsu '
alias uninsy='yay -Rsu'
alias update='sudo pacman -Syu'
alias updatey='yay -Syu --noconfirm'
alias unlock='sudo rm /var/lib/pacman/db.lck'    # remove pacman lock
alias tlmgr='tllocalmgr install' 

# Colorize grep output (good for log files)
alias grep='grep --color=auto'
alias egrep='egrep --color=auto'
alias fgrep='fgrep --color=auto'

# flags 
alias df='df -h'                          # human-readable sizes
alias free='free -m'                      # show sizes in MB

## get top process eating memory
alias psmem='ps auxf | sort -nr -k 4'
alias psmem10='ps auxf | sort -nr -k 4 | head -10'

## get top process eating cpu ##
alias pscpu='ps auxf | sort -nr -k 3'
alias pscpu10='ps auxf | sort -nr -k 3 | head -10'

# youtube-dlp
alias ytab="youtube-dlp --extract-audio --audio-format best "
alias ytall="youtube-dlp --extract-audio --audio-format flac "
alias ytam="youtube-dlp --extract-audio --audio-format mp3 "
alias ytv="youtube-dlp -f bestvideo+bestaudio "

# Useful curls.
alias rr='curl -s -L https://bit.ly/2VRgukx | bash'

# shortcuts
alias r='clear; sudo -s'
alias nf='c;neofetch'
alias cm='cmatrix -Bras'
alias btime='systemd-analyze'
alias btimeb='systemd-analyze blame'
alias btimecc='systemd-analyze critical-chain'
alias det='pacman -Qi'
alias nu='vnstat -d 1'
alias listpkg='expac -H M "%011m\t%-20n\t%10d" $(comm -23 <(pacman -Qqen | sort) <({ pacman -Qqg xorg; expac -l '\n' '%E' base; } | sort -u)) | sort -n'

# timer script
timer() {
        local N=$1; shift
        (sleep $N && mpg123 -q /path/to/audio/alert.mp3) &&
        echo "timer set for $N"
}

[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh