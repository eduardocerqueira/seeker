#date: 2023-06-21T17:09:00Z
#url: https://api.github.com/gists/65f53cf0c773e2cfd11c738b314e6d72
#owner: https://api.github.com/users/danielcarr

# SET PROMPT
# $' ... ' is necessary to include control characters, eg \e
# %{ ... %} encloses colours to indicate that it shouldn't count towards line length
#
# \e[2m = dimmed
# %n = username
#
# \e[0;37m = normal, white
# @ = literal @
#
# \e[93m = bright yellow
# %~ = current directory
# 
# \e[2;36 = dim cyan/teal (it comes before the command to make sure the control escaping works)
# $(`command -v __git_ps1 >/dev/null` && __git_ps1 " [%s]") = if git-prompt is available, insert the git status when in a git repository
# " [%s]" = the format of the git status; status (eg branch) preceded by a space and surrounded by brackets (the quotation marks are to keep the space there)
# \e[0m = reset to normal after coloured git status so that the colour can be set to white (it comes after the command to make sure command escaping works)
# 
# \e[1;97m = bold, bright white
# %# = The prompt symbol, % for normal user, # for sudo
# \e[0m = Reset to default colours for the user input part of the prompt
# End with a space before user input
export PROMPT=$'%{\e[2m%}%n%{\e[0;37m%}@%{\e[93m%}%~%{\e[2;36m%}$(`command -v __git_ps1 >/dev/null` && __git_ps1 " [%s]")%{\e[0m%}%{\e[1;97m%}%#%{\e[0m%} '

# RIGHT SIDE PROMPT
# A (bold) green tick if the last command succeeded, else the status code in red (and bolded), followed by the time in grey and underlined
export RPROMPT='%B%(?.%F{green}%1{✓%}%f.%F{red}%?%f)%b %F{8}%U%T%u%f'

# Enable command substitution in the prompt
setopt promptsubst
