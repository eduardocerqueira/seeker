#date: 2021-09-24T16:59:33Z
#url: https://api.github.com/gists/7fb615c8928293695c310a97aecf5179
#owner: https://api.github.com/users/carolgilabert

# path to oh my zsh
export ZSH="/Users/carolgilabert/.oh-my-zsh"

# theme
ZSH_THEME="emoji-pi"

# plugins
plugins=(git zsh-autosuggestions zsh-syntax-highlighting colored-man-pages nvm osx npm)

# load oh my zsh stuff
source $ZSH/oh-my-zsh.sh

# aliases
alias zshconfig="code ~/.zshrc"
alias ohmyzsh="code ~/.oh-my-zsh"
alias dev="npm run dev"
alias gcm="git checkout main"

# loading nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# yarn stuff?
export PATH="$HOME/.yarn/bin:$HOME/.config/yarn/global/node_modules/.bin:$PATH"

# any commands run with a space in front won't be saved to history
setopt HIST_IGNORE_SPACE
