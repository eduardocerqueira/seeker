#date: 2022-06-27T16:56:08Z
#url: https://api.github.com/gists/bae91c3d8968eca7f7762cbe79b72c16
#owner: https://api.github.com/users/donnfelker

# Hat tip to Kaushik Gopal for some of this 

# make zsh tab completion fix capitalization errors for directories and files
# i don't know if this is required anymore
# autoload -Uz compinit && compinit

# 0 -- vanilla completion (abc => abc)
# 1 -- smart case completion (abc => Abc)
# 2 -- word flex completion (abc => A-big-Car)
# 3 -- full flex completion (abc => ABraCadabra)
zstyle ':completion:*' matcher-list '' \
  'm:{a-z\-}={A-Z\_}' \
  'r:[^[:alpha:]]||[[:alpha:]]=** r:|=* m:{a-z\-}={A-Z\_}' \
  'r:|?=** m:{a-z\-}={A-Z\_}'


# required for homebrew on M1s
eval "$(/opt/homebrew/bin/brew shellenv)"
eval "$(starship init zsh)" 


# autcomplete customizations
autoload -U compinit
compinit
compdef g=git

# Edit Properties files of zsh, git, gradle
alias zshpro="code ~/.zshrc;. ~/.zshrc"
alias gitconfig="code ~/.gitconfig; ~/.gitconfig"
alias gradleprops="code ~/.gradle/gradle.properties"

alias g="git"
alias o="open ."
alias la="ls -Gla"
alias mv="mv -i" # Cause mv to write a prompt to standard error before moving a file that would overwrite an existing file.

# Restart ADB server
alias adbr="adb kill-server; adb start-server"


# Clean up all squashed and merged branches locally
alias gclean='git checkout -q main && git for-each-ref refs/heads/ "--format=%(refname:short)" | while read branch; do mergeBase=$(git merge-base main $branch) && [[ $(git cherry main $(git commit-tree $(git rev-parse "$branch^{tree}") -p $mergeBase -m _)) == "-"* ]] && git branch -D $branch; done'


androidBuildToolsVersion() {
  echo $ANDROID_BUILD_TOOLS_VERSION
}

# list the most recent files in a directory
lsnew()
{
	ls -lt ${1+"$@"} | head -20;
}

# open a manpage in Preview, which can be saved to PDF
pman()
{
  # Old versions of Mac preview is here: /Applications/Preview.app
  man -t "${1}" | open -f -a /System/Applications/Preview.app 
}

export EDITOR="code --wait"
export ANDROID_HOME="$HOME/Library/Android/sdk"
export ANDROID_TOOLS=$ANDROID_HOME/tools
export ANDROID_BUILD_TOOLS_VERSION=`ls $ANDROID_HOME/build-tools | sort -r | head -n 1`
PATH="$ANDROID/HOME/tools:$ANDROID_HOME/tools/bin:$ANDROID_HOME/platform-tools:$PATH"


#THIS MUST BE AT THE END OF THE FILE FOR SDKMAN TO WORK!!!
export SDKMAN_DIR="$HOME/.sdkman"
[[ -s "$HOME/.sdkman/bin/sdkman-init.sh" ]] && source "$HOME/.sdkman/bin/sdkman-init.sh"

# has to be at the very end
source /opt/homebrew/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
source /opt/homebrew/share/zsh-autosuggestions/zsh-autosuggestions.zsh
