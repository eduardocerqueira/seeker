#date: 2022-04-15T17:12:48Z
#url: https://api.github.com/gists/f72009ee78c4d2517c097f2483c55b9d
#owner: https://api.github.com/users/tz4678

#!/usr/bin/zsh

source /usr/share/zsh/scripts/zplug/init.zsh

# набор хелперов
zplug "robbyrussell/oh-my-zsh", as:plugin, use:"lib/*.zsh"

# Автоматическая подстановка парных кавычек и скобок
zplug "hlissner/zsh-autopair", defer:2
# Не работает когда asdf установлен не в домашний каталог
#zplug "plugins/asdf", from:oh-my-zsh
# Алиасы для пакетных менеджеров
zplug "plugins/archlinux", from:oh-my-zsh
zplug "plugins/command-not-found", from:oh-my-zsh
# ^O - копировать текущую строку в буффер
zplug "plugins/copybuffer", from:oh-my-zsh
zplug "plugins/copydir", from:oh-my-zsh
zplug "plugins/copyfile", from:oh-my-zsh
zplug "plugins/docker", from:oh-my-zsh
zplug "plugins/docker-compose", from:oh-my-zsh
zplug "plugins/dotenv", from:oh-my-zsh
zplug "plugins/extract", from:oh-my-zsh
zplug "plugins/fzf", from:oh-my-zsh
zplug "plugins/git", from:oh-my-zsh
zplug "plugins/git-flow", from:oh-my-zsh
zplug "plugins/history", from:oh-my-zsh
zplug "plugins/history-substring-search", from:oh-my-zsh
zplug "plugins/sudo", from:oh-my-zsh
zplug "zsh-users/zsh-autosuggestions"
zplug "zsh-users/zsh-completions"
zplug "zdharma/fast-syntax-highlighting"
zplug "plugins/web-search", from:oh-my-zsh
zplug "MichaelAquilina/zsh-you-should-use"

# Install plugins if there are plugins that have not been installed
if ! zplug check --verbose; then
  printf "Install? [y/N]: "
  if read -q; then
    echo; zplug install
  fi
fi

zplug load

eval "$(starship init zsh)"

typeset -U path
path+=(~/.local/bin ~/bin)
fpath+=~/.zfunc

# Переменные окружения, независимые от оболочки, следует добавлять в ~/.config/environment.d/envvars.conf.

# размер истории
export HISTSIZE=1000000
export SAVEHIST=$HISTSIZE

# настройки регистронезависимы, подчеркивания вырезаются (autoCD=auto_cd)
# Для отключения какой-то настройки к имени нужно добавить no
# см. вывод setopt, который без аргументов показывает установленные настройки
# setopt autocd
setopt auto_list # automatically list choices on ambiguous completion
setopt auto_menu # automatically use menu completion
#setopt complete_in_word
#setopt correct_all
setopt correct
setopt interactive_comments # allow comments in interactive shells
# включаем использование регулярных выражений с *
setopt extended_glob

# шарим историю между сессиями
#setopt share_history
# append to history
setopt append_history
# adds commands as they are typed, not at shell exit
setopt inc_append_history
# добавлять метку времени к имени команды в истории
setopt extended_history
# не сохраняем команду history в истории
setopt hist_no_store
# убираем лишние пробелы
setopt hist_reduce_blanks
# убираем повторы при поиске по истории
setopt hist_find_no_dups

alias cp='cp --reflink=auto --sparse=always'
alias vim=micro
alias reload="exec -l $SHELL"
alias zshrc="$EDITOR $HOME/.zshrc && source $HOME/.zshrc"
# alias vimrc="vim $MYVIMRC"
alias ws="cd $HOME/workspace"
alias q='exit'

# fix: yay ставит некоторые пакеты в ~/.asdf/installs/python...
alias yay='env PATH="${PATH//~\/.asdf\/shims:/}" yay'

mkcd() { mkdir -p "$1" && cd "$1" }

# if [ $TILIX_ID ] || [ $VTE_VERSION ]; then
#   source /etc/profile.d/vte.sh
# fi

if [ -d /opt/asdf-vm ]; then
  export ASDF_DIR=/opt/asdf-vm
  source $ASDF_DIR/asdf.sh
fi

function pet-select() {
  BUFFER=$(pet search --query "$LBUFFER")
  CURSOR=$#BUFFER
  zle redisplay
}

zle -N pet-select
stty -ixon
bindkey '^s' pet-select

#neofetch
