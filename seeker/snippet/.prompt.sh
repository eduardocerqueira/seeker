#date: 2022-04-25T17:15:33Z
#url: https://api.github.com/gists/5fcc2b75d622830a91cd5189596be461
#owner: https://api.github.com/users/jhaisley

#!/bin/bash
#source in bashrc

set_prompt()
{
   local last_cmd=$?
   local txtreset='$(tput sgr0)'
   local txtbold='$(tput bold)'
   local txtblack='$(tput setaf 0)'
   local txtred='$(tput setaf 1)'
   local txtgreen='$(tput setaf 2)'
   local txtyellow='$(tput setaf 3)'
   local txtblue='$(tput setaf 4)'
   local txtpurple='$(tput setaf 5)'
   local txtcyan='$(tput setaf 6)'
   local txtwhite='$(tput setaf 7)'
   # unicode "✗"
   local fancyx='\342\234\227'
   # unicode "✓"
   local checkmark='\342\234\223'
   # Line 1: Full date + full time (24h)
   # Line 2: current path
   PS1="\[$txtbold\]\[$txtwhite\]\n\D{%A %d %B %Y %H:%M:%S}\n\[$txtgreen\]\w\n"
   # User color: red for root, yellow for others
   if [[ $EUID == 0 ]]; then
       PS1+="\[$txtred\]"
   else
       PS1+="\[$txtyellow\]"   
   fi
   # Line 3: user@host
   PS1+="\u\[$txtwhite\]@\h\n"
   # Line 4: a red "✗" or a green "✓" and the error number
   if [[ $last_cmd == 0 ]]; then
      PS1+="\[$txtgreen\]$checkmark \[$txtwhite\](0)"
   else
      PS1+="\[$txtred\]$fancyx \[$txtwhite\]($last_cmd)"
   fi
   # Line 4: good old prompt, $ for user, # for root
   PS1+=" \\$ "
}
PROMPT_COMMAND='set_prompt'