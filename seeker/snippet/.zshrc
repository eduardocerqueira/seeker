#date: 2023-02-09T16:58:49Z
#url: https://api.github.com/gists/202fc38104c1ce21cfb2912cb681bd91
#owner: https://api.github.com/users/efcor

# add sbin to path
export PATH="/usr/local/sbin:$PATH"

# for these homebrew php's below: to use a version (only on command line- valet takes more steps)
# uncomment the lines below the version to use and make sure the rest of the versions are commented

# homebrew php 7.4
#export PATH="/usr/local/opt/php@7.4/bin:$PATH" 
#export PATH="/usr/local/opt/php@7.4/sbin:$PATH"

# homebrew php 8.1
export PATH="/usr/local/opt/php@8.1/bin:$PATH" 
export PATH="/usr/local/opt/php@8.1/sbin:$PATH"

# homebrew php latest (8.2 as of 12/16/2022)
#export PATH="/usr/local/opt/php/bin:$PATH" 
#export PATH="/usr/local/opt/php/sbin:$PATH"

# composer bin
export PATH="$HOME/.composer/vendor/bin:$PATH"

# directory shortcuts
alias desk='cd ~/Desktop'

# command aliases
alias c='clear'
alias l='ls -lah'
alias profile='edit ~/.zshrc'
alias refresh='source ~/.zshrc'
alias u='vendor/bin/phpunit'
function edit { code "${1:-.}" } # open specified file/dir in vscode; default: open current dir
dir () { mkdir "$1"; cd "$1"; } # create specified dir and cd into it
nf () { find "${1:-.}" -type f | wc -l; } # get number of files in a dir recursively
sc () { sc-dl --url "https://soundcloud.com/$1" --dir ~/Desktop  }
setphp81 () { ./setphpcli81.sh && source .zshrc && valet use php@8.1 --force }
setphp74 () { valet use php@7.4 --force && ./setphpcli74.sh && source .zshrc }
setphpcli81 () { ./setphpcli81.sh && source .zshrc }
setphpcli74 () { ./setphpcli74.sh && source .zshrc }

# git shortcuts
alias fixgitwork='git config user.name "Bob Smith (Replace this" && git config user.email "example@work.edu (replace this)"'
alias fixgitpers='git config user.name "Bob Smith (Replace this)" && git config user.email "example@gmail.com (replace this)"'

# ssh shortcuts
alias someserver='ssh someserver'

# shell colors
export CLICOLOR=1
export LSCOLORS=gxBxhxDxfxhxhxhxhxcxcx

========= Contents of setphpcli74.sh ============
# uncomment php 7.4
sed -i '' 's|^#export PATH="/usr/local/opt/php@7.4/bin:$PATH"|export PATH="/usr/local/opt/php@7.4/bin:$PATH"|' .zshrc
sed -i '' 's|^#export PATH="/usr/local/opt/php@7.4/sbin:$PATH"|export PATH="/usr/local/opt/php@7.4/sbin:$PATH"|' .zshrc

# comment out the other php versions (the "^" keeps from bothering ones that are already commented out)
#7.3
sed -i '' 's|^export PATH="/usr/local/opt/php@7.3/bin:$PATH"|#export PATH="/usr/local/opt/php@7.3/bin:$PATH"|' .zshrc
sed -i '' 's|^export PATH="/usr/local/opt/php@7.3/sbin:$PATH"|#export PATH="/usr/local/opt/php@7.3/sbin:$PATH"|' .zshrc
#8.1
sed -i '' 's|^export PATH="/usr/local/opt/php@8.1/bin:$PATH"|#export PATH="/usr/local/opt/php@8.1/bin:$PATH"|' .zshrc
sed -i '' 's|^export PATH="/usr/local/opt/php@8.1/sbin:$PATH"|#export PATH="/usr/local/opt/php@8.1/sbin:$PATH"|' .zshrc
#latest
sed -i '' 's|^export PATH="/usr/local/opt/php/bin:$PATH"|#export PATH="/usr/local/opt/php/bin:$PATH" |' .zshrc
sed -i '' 's|^export PATH="/usr/local/opt/php/sbin:$PATH"|#export PATH="/usr/local/opt/php/sbin:$PATH"|' .zshrc

========= Contents of setphpcli81.sh ============
# uncomment php 8.1
sed -i '' 's|^#export PATH="/usr/local/opt/php@8.1/bin:$PATH"|export PATH="/usr/local/opt/php@8.1/bin:$PATH"|' .zshrc
sed -i '' 's|^#export PATH="/usr/local/opt/php@8.1/sbin:$PATH"|export PATH="/usr/local/opt/php@8.1/sbin:$PATH"|' .zshrc

# comment out the other php versions (the "^" keeps from bothering ones that are already commented out)
#7.3
sed -i '' 's|^export PATH="/usr/local/opt/php@7.3/bin:$PATH"|#export PATH="/usr/local/opt/php@7.3/bin:$PATH"|' .zshrc
sed -i '' 's|^export PATH="/usr/local/opt/php@7.3/sbin:$PATH"|#export PATH="/usr/local/opt/php@7.3/sbin:$PATH"|' .zshrc
#7.4
sed -i '' 's|^export PATH="/usr/local/opt/php@7.4/bin:$PATH"|#export PATH="/usr/local/opt/php@7.4/bin:$PATH"|' .zshrc
sed -i '' 's|^export PATH="/usr/local/opt/php@7.4/sbin:$PATH"|#export PATH="/usr/local/opt/php@7.4/sbin:$PATH"|' .zshrc
#latest
sed -i '' 's|^export PATH="/usr/local/opt/php/bin:$PATH"|#export PATH="/usr/local/opt/php/bin:$PATH" |' .zshrc
sed -i '' 's|^export PATH="/usr/local/opt/php/sbin:$PATH"|#export PATH="/usr/local/opt/php/sbin:$PATH"|' .zshrc

