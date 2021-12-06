#date: 2021-12-06T17:08:38Z
#url: https://api.github.com/gists/fd29b54de6423448af274e77437e23b5
#owner: https://api.github.com/users/cereda

# directory to store all antigen stuff -- optional,
# I just like to keep things out of my home dir
ADOTDIR=/opt/paulo/applications/antigen/payload

# source to the antigen script, obtained from
# curl -L git.io/antigen > antigen.zsh
source /opt/paulo/applications/antigen/antigen.zsh

# tell antigen to use the omz framework
antigen use oh-my-zsh

# set the theme, antigen will fetch it for us
antigen theme spaceship-prompt/spaceship-prompt

# these are plugins from omz
antigen bundle git
antigen bundle colored-man-pages
antigen bundle colorize
antigen bundle common-aliases
antigen bundle copyfile

# these are custom plugins from GitHub repos
antigen bundle zsh-users/zsh-syntax-highlighting
antigen bundle zsh-users/zsh-autosuggestions

# then I simply apply antigen
antigen apply

# my general omz customization
CASE_SENSITIVE="true"
DISABLE_AUTO_UPDATE="true"
COMPLETION_WAITING_DOTS="true"

# this is for the spaceship theme -- I like
# to have those shown in my terminal
SPACESHIP_USER_SHOW=always
SPACESHIP_HOST_SHOW=always

# fortune cookies, yay!
if [ -f /usr/bin/fortune ]; then
    /usr/bin/fortune
fi

# to update everything (theme and plugins)
# just run antigen update
