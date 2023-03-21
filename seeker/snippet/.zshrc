#date: 2023-03-21T16:44:21Z
#url: https://api.github.com/gists/ff1d8655acafbb37152677ed661d67fc
#owner: https://api.github.com/users/FractalHQ

# Fig pre block. Keep at the top of this file.
[[ -f "$HOME/.fig/shell/zshrc.pre.zsh" ]] && builtin source "$HOME/.fig/shell/zshrc.pre.zsh"
DEFAULT_USER=$USER

# Enable Powerlevel10k instant prompt. Should stay close to the top of ~/.zshrc.
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# Path to your oh-my-zsh installation.
export ZSH="/Users/fractal/.oh-my-zsh"

## Active Terminal Theme ##
ZSH_THEME=powerlevel10k/powerlevel10k
# ZSH_THEME=muse
# ZSH_THEME=spaceship

## OhMyZSH Plugins ##
plugins=(git macos last-working-dir zsh-autosuggestions fzf zsh-navigation-tools)

source $ZSH/oh-my-zsh.sh

## Powerline Fonts ##
# . ~/Library/Python/3.8/lib/python/site-packages/powerline/bindings/zsh/powerline.zsh

## Aliases ##
alias o="open ./"	# open current directory in Finder
alias emptytrash="trash ~/.Trash"
alias pwdc="pwd | tr -d '\n' | pbcopy && echo 'pwd copied to clipboard'"	# copy current path to clipboard
alias yg="yarn global"
alias pg="sudo -u postgres -i && postgresql"	# activate postgresql
alias blender="~/../../Applications/Blender/3.1/Blender_3.1.2.app/Contents/MacOS/Blender"
alias gs="git status"
alias c="code-insiders ./"
alias zshrc="code-insiders ~/.zshrc"
alias hax="code-insiders ~/.zshrc"		# open this vile in vscode
alias haxx="code-insiders ~/.p10k.zsh"	# open .p10k config in vscode
alias reload="source ~/.zshrc"
alias ohmyzsh="code-insiders ~/.oh-my-zsh"
alias k8="kubectl"			# devops engineer roleplay
alias eyes="glances"		# run glances (process viewer)
alias tc="tree -c -L 1 -C"	# tree level 1	↓ date modified
alias t="tree -L 1 -C"		# tree level 1
alias t2="tree -L 2 -C"		# tree level 2
alias t3="tree -L 3 -C"		# tree level 3
alias t4="tree -L 4 -C"		# tree level 4
alias gppm="git push production main" # push to production
alias gpp="git pull production main; git push; echo '\n✅ Pulled main from production and pushed main to origin.\n'" # pull production and push origin
alias kys="killall SpeechSynthesisServer"  # kill macos voice
alias show="defaults write com.apple.finder AppleShowAllFiles TRUE; killall Finder" 	# show hidden files (MacOS)
alias hide="defaults write com.apple.finder AppleShowAllFiles FALSE; killall Finder" 	# hide hidden files (MacOS)
alias awesome="awesome-hub"
alias yt="youtube"
alias ports="sudo lsof -i -P | grep LISTEN" # List open ports and their process id
alias fracflix="echo '

🤖 FracBOT initializing FracFLIX....


    .,,,,,,,,,,,,.*,,,,,,,,,,. ,       *,,,,,,,,*      *.,,,,,,,,.,  . ,,,,,,,,,,,,,..,,,,,*       *,,,,,.  ,,,,,.    . ,,,,
    ,************. *************,*    . ******** .   . ************...,************, ,*****         *****,. *****,*  ..*****
    ,************, **************,*   ..********..    ******..,*****...************, ,*****         *****,.. *****.   *****,.
    ,****,.        *****,. ,,*****     **********   ..***** . .,****,.,*****        .,*****         *****,. .,***** .,****,.
    ,****,.        *****,. .,*****   ..**** ,****.. . ***** .  ,****...*****        .,*****         *****,.  .*****,.*****
    ,****,.        *****,. ..*****   *,**** ,****,* . ***** . ..****...*****        .,*****         *****,.  . ********** .
    ,****,,. ....  *****..,,***** .   ***** ******  ..***** .        ..*********** ..******         *****,    .,********,.
    ,**********,   ************,,.  ..****,* *****, . ***** .        ..*********** ..,*****         *****,.    .,*******
    ,**********,   **************.  .,**** . ,****, . ***** . ..,,,,..****** ...... .,*****         *****,.   ..********.
    ,****...,,,,   *****,,* ,*****   *****.,,,***** . ***** . .,****,.,*****        .******         *****,.  ..**********
    ,****,.        *****,. .,***** .,**************..****** . .,****,..*****        .,*****         *****,.  .***** *****,,
    ,****,.        *****,. .,*****  ,********,******  *****, ,.*****,*,*****        .,***** *.      *****,. ,*****,. *****..
    ,****,.        *****,. .,*****  *****    .****** ,,************,..,*****        .,************* *****,...***** ..,****,
    ,****,.        *****,. .,***** ,****,.     *****... ,*********.. .,*****        ..************* *****,. *****,* . *****..
    ,****,.        *****,* ..**,,,.......      ,,,*..    .,,,,,,.         ..         ,,........,,,,.*****, ,****,    *,*****
    ,****,*       *,,,,.                                                                                   ,,,....    .*****,
     ,,,.                                                                                                               .,,,


' && cliflix"
## note: made with cflix https://github.com/fabiospampinato/cliflix (creator of Notable XD)
## note: ascii made with https://manytools.org/hacker-tools/convert-images-to-ascii-art/
## note: netflix font made with https://www.fontmeme.com

alias settings="ci /Users/fractal/Library/Application\ Support/Code\ -\ Insiders/User"

alias ip="ipconfig getifaddr en0"   # print ip address
alias yd="yarn install && yarn dev"
alias i="pnpm install --save --"
alias is="pnpm install --save-dev --"
alias d="pnpm run dev"
alias pd="pnpm dev"
# alias ps="pnpm start"
alias pb="pnpm build"
alias pp="pnpm package"
alias ppr="pnpm preview"
alias pt="pnpm test"
alias pc="pnpm check"
alias ptui="pnpm test:ui"
alias bump="pnpm up --config.strict-peer-dependencies=false --latest -r"
## Trash & Dev (pnpm)
alias ry="trash node_modules yarn.lock package-lock.json pnpm-lock.yaml dist public/build && yarn && yarn dev"
## "ReDev" (pnpm)
alias rd="trash node_modules yarn.lock package-lock.json pnpm-lock.yaml dist && pnpm install && pnpm run dev"
## "ReBuild" (pnpm)
alias rb="trash node_modules pnpm-lock.yaml yarn.lock package-lock.json dist && pnpm install && pnpm run build"
alias psort="pnpx sort-package-json"
alias ts="pnpx typesync"
## Trash all dev directories
alias ta="trash .pnpm-debug.log pnpm-lock.yaml node_modules build dist .svelte-kit .vercel"

# Kills all running vite servers (for when vscode fails to kill them and they're orphaned)
function yeetVite() {
  # Find any vite processes
  viteProcesses=$(ps aux | grep -v grep | grep -i vite | awk '{print $2}')
  echo "viteProcesses: $viteProcesses"
  
  # If there are no vite processes, exit
  if [ -z "$viteProcesses" ]; then
    echo "No vite processes found"
    return
  fi
  
  # Yeet them all
  for viteProcess in $viteProcesses; do
    echo "Yeeting vite process $viteProcess"
    kill -9 $viteProcess > /dev/null 2>&1
  done
  
  echo "vite has been yeeted 👍"
}

## Spotify Controls ##
alias spot="spotify status"
alias play="spotify play"
alias stop="spotify stop"

##? Spyware Zapper ##
alias fuckadobe='
echo "Starting 🔎";
echo "Killing Adobe Bullshit";
sudo pkill "Adobe Desktop Service";
sudo pkill "AdobeCRDaemon";
sudo pkill "AdobeIPCBroker";
sudo pkill "CCXProcess";
sudo pkill "Core Sync";
sudo pkill "Core Sync Helper";
sudo pkill "Creative Cloud Helper";
echo "Exiting 👋🏽";'

alias prc='
echo -e """{\n\t\"svelteSortOrder\": \"scripts-markup-styles\",\n\t\"htmlWhitespaceSensitivity\": \"ignore\",\n\t\"trailingComma\": \"all\",\n\t\"requirePragma\": false,\n\t\"bracketSpacing\": true,\n\t\"singleQuote\": true,\n\t\"printWidth\": 100,\n\t\"useTabs\": false,\n\t\"tabWidth\": 4,\n\t\"semi\": true\n}""" >> .prettierrc;
echo "Created .prettierrc 🌷";
cat .prettierrc;
'
alias ci="code-insiders"

alias hb="sh ~/dev/scripts/shell/hb.sh"

####?
####? Web Search https://github.com/ohmyzsh/ohmyzsh/blob/master/plugins/web-search/web-search.plugin.zsh
####?

function web_search() {
  emulate -L zsh

  # define search engine URLS
  typeset -A urls
  urls=(
    # $ZSH_WEB_SEARCH_ENGINES
    google      "https://www.google.com/search?q="
    # bing        "https://www.bing.com/search?q="
    duckduckgo  "https://www.duckduckgo.com/?q="
    github      "https://github.com/search?q="
    stackoverflow  "https://stackoverflow.com/search?q="
    # wolframalpha   "https://www.wolframalpha.com/input/?i="
  )

  # check whether the search engine is supported
  if [[ -z "$urls[$1]" ]]; then
    echo "Search engine '$1' not supported."
    return 1
  fi

  # search or go to main page depending on number of arguments passed
  if [[ $# -gt 1 ]]; then
    # build search url:
    # join arguments passed with '+', then append to search engine URL
    url="${urls[$1]}${(j:+:)@[2,-1]}"
  else
    # build main page url:
    # split by '/', then rejoin protocol (1) and domain (2) parts with '//'
    url="${(j://:)${(s:/:)urls[$1]}[1,2]}"
  fi

  open_command "$url"
}

alias google='web_search google'
alias ddg='web_search duckduckgo'
alias github='web_search github'
alias so='web_search stackoverflow'

# ddg bangs
alias wiki='web_search duckduckgo \!w'
alias news='web_search duckduckgo \!n'
alias youtube='web_search duckduckgo \!yt'
alias map='web_search duckduckgo \!m'
alias image='web_search duckduckgo \!i'
alias ducky='web_search duckduckgo \!'

# View the latest release notes for a GitHub repo
function changelog() {
  if [[ -z "$1" ]]; then
    echo "Usage: changelog <repo>"
    return 1
  fi

  gh release view -R $1
}

# Open a GitHub repo in the browser
function ghw() {
  if [[ -z "$1" ]]; then
    echo "Usage: ghw <repo>"
    return 1
  fi

  gh repo view $1 --web
}

# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh.
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh

# To activate the syntax highlighting, add the following at the end of your .zshrc:
source /usr/local/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh

# . /usr/local/opt/asdf/asdf.sh

test -e "${HOME}/.iterm2_shell_integration.zsh" && source "${HOME}/.iterm2_shell_integration.zsh"

[[ -s /Users/fractal/.autojump/etc/profile.d/autojump.sh ]] && source /Users/fractal/.autojump/etc/profile.d/autojump.sh

# autoload -U compinit && compinit -u

function p10k-on-pre-prompt() {
  if (( COLUMNS < 80 )); then
    p10k display '*/context'=hide
  else
    p10k display '*/context'=show
  fi
}

export PATH=/usr/local/opt/python/libexec/bin:$PATH

alias http="httpstat"
export HTTPSTAT_SHOW_IP=false
export HTTPSTAT_SHOW_SPEED=true
export HTTPSTAT_SAVE_BODY=false

export VOLTA_HOME="$HOME/.volta"
export PATH="$VOLTA_HOME/bin:$PATH"

export PNPM_HOME="/Users/fractal/Library/pnpm"
export PATH="$PNPM_HOME:$PATH"

# bun completions
[ -s "/Users/fractal/.bun/_bun" ] && source "/Users/fractal/.bun/_bun"

# Bun
export BUN_INSTALL="/Users/fractal/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

# pnpm
export PNPM_HOME="/Users/fractal/Library/pnpm"
export PATH="$PNPM_HOME:$PATH"
# pnpm end

# Fig post block. Keep at the bottom of this file.
[[ -f "$HOME/.fig/shell/zshrc.post.zsh" ]] && builtin source "$HOME/.fig/shell/zshrc.post.zsh"
export PATH="/usr/local/sbin:$PATH"
