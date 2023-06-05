#date: 2023-06-05T17:07:24Z
#url: https://api.github.com/gists/69cfc7a2688dea70060e29d06db9b147
#owner: https://api.github.com/users/mattbey-dsw

##############################################################
###  PROMPT CONFIGURATION                                  ###
##############################################################

# Enable Powerlevel10k instant prompt. Should stay close to the top of ~/.zshrc.
# Initialization code that may require console input (password prompts, [y/n]
# confirmations, etc.) must go above this block; everything else may go below.
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# Disable insecure directories from being loaded by oh-my-zsh
# due to not being admins on the machine and not having access
# to those system directories
ZSH_DISABLE_COMPFIX=true

# Path to your oh-my-zsh installation.
export ZSH="/Users/mb450500/.oh-my-zsh"

# Set name of the theme to load.
ZSH_THEME="powerlevel10k/powerlevel10k"

# Uncomment the following line to use case-sensitive completion.
# CASE_SENSITIVE="true"

# Uncomment the following line to use hyphen-insensitive completion.
# Case-sensitive completion must be off. _ and - will be interchangeable.
HYPHEN_INSENSITIVE="true"

# Which plugins would you like to load?
# Standard plugins can be found in $ZSH/plugins/
# Custom plugins may be added to $ZSH_CUSTOM/plugins/
plugins=(git zsh-syntax-highlighting)

source $ZSH/oh-my-zsh.sh

# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh.
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh

##############################################################
###  USER CONFIGURATION                                    ###
##############################################################

# Unset LESS to prevent repaint when exiting alternate screen content
unset LESS;

# CLOUD ALIASES
dbtoken() {
  az account get-access-token --resource-type oss-rdbms --query accessToken | sed 's/\"//g'
}

# MISC. ALIASES
search_rfk_feed() {
	grep -e ^product_id -e ^$1 $2
}

# NPM ALIASES
alias ni='npm install'
alias nid='npm install --save-dev'
alias nig='npm install --global'
alias nt='npm test'
alias nit='npm install && npm test'
alias nk='npm link'
alias nr='npm run'
alias ns='npm start'
alias nstart='ns'
alias nserve='npm run serve'
alias nsv='npm run serve'
alias nclean='rm -rf node_modules && npm cache clean --force'
alias nf='npm cache clean && rm -rf node_modules && npm install'
alias nls='npm ls'
alias nlsg='npm list --global --depth=0'

npm_update_deps() {
	npm update
	npm list --json | jq --slurpfile package package.json '
	def replaceVersion($replacements):
		with_entries(
			if .value | startswith("^")
			then
				.value = ("^" + $replacements[.key].version)
			else
				.
			end
		);

	.dependencies as $resolved
	| reduce ["dependencies", "devDependencies"][] as $deps (
		$package[0];
		if .[$deps] | type == "object"
		then
			.[$deps] |= replaceVersion($resolved)
		else
			.
		end
	)' > package.json~
	mv package.json~ package.json
	npm install
}

##############################################################
###  TOOLING CONFIGURATION                                 ###
##############################################################

###  DBI SETTINGS  ###########################################

### NETSKOPE: Use custom enterprise cert for NetSkope  #######
# node.js // npm
export NODE_EXTRA_CA_CERTS="/Library/Application Support/Netskope/STAgent/download/nscacert.pem"
# Azure CLI: doesn't appear to be needed anymore
# export REQUESTS_CA_BUNDLE="/Library/Application Support/Netskope/STAgent/download/nscacert.pem"

###  DEV TOOL SETTINGS  ######################################

# Add Visual Studio Code (code)
export PATH="$PATH:/Applications/Visual Studio Code.app/Contents/Resources/app/bin"

# DotNet Setup
export DOTNET_ROOT="/opt/homebrew/opt/dotnet/libexec"
export PATH=$PATH:$DOTNET_ROOT

# NVM configuration
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# The next line updates PATH for the Google Cloud SDK.
if [ -f '/Users/mb450500/google-cloud-sdk/path.zsh.inc' ]; then . '/Users/mb450500/google-cloud-sdk/path.zsh.inc'; fi

# The next line enables shell command completion for gcloud.
if [ -f '/Users/mb450500/google-cloud-sdk/completion.zsh.inc' ]; then . '/Users/mb450500/google-cloud-sdk/completion.zsh.inc'; fi

source /Users/mb450500/.docker/init-zsh.sh || true # Added by Docker Desktop

# SDKMAN configuration 
export SDKMAN_DIR="$HOME/.sdkman"
[[ -s "$HOME/.sdkman/bin/sdkman-init.sh" ]] && source "$HOME/.sdkman/bin/sdkman-init.sh"
