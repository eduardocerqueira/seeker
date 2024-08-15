#date: 2024-08-15T16:53:32Z
#url: https://api.github.com/gists/41526c2957513e6392555309dd435c4c
#owner: https://api.github.com/users/ppcamp


# thanks to https://github.com/junegunn/fzf/issues/2789#issuecomment-2196524694
rgfzf() {
  if [ ! "$#" -gt 0 ]; then echo "Need a string to search for!"; return 1; fi

  rg --color=always --line-number --no-heading --smart-case "#{*:-}" \
  | fzf -d':' --ansi \
      --preview "bat -p --color=always {1} --highlight-line{2}" \
      --preview-window ~8,+{2}-5 \
  | awk -F':' '{ print $1 ":" $2 }' \
  | xargs -r -I {} code -g {}
}

# fif shows the file that matched with the search, and using fzf
# allows user to see it, similar behavior to 'telescope' nvim ext.
#
# You can also open vscode in the selected line, in the end
#
# thanks to https://www.reddit.com/r/commandline/comments/fu6zzp/search_file_content_with_fzf/
# thanks to https://github.com/junegunn/fzf/issues/2789#issuecomment-2196524694
fif() {
  if [ ! "$#" -gt 0 ]; then echo "Need a string to search for!"; return 1; fi

  rg --line-number --no-heading --smart-case "#{*:-}" \
  | awk -F: '{ printf "\033[1;32m%s\033[0m:\033[1;34m%s\033[0m\n", $1, $2 }' \
  | fzf -d':' --ansi \
      --preview "bat -p --color=always {1} --highlight-line{2}" \
      --preview-window ~8,+{2}-5 \
  | xargs -r -I {} code -g {}
}


# Define the widget to execute your functino
# thanks to https://superuser.com/a/1564526
#
# Bind Ctrl+X to open a search box that will allow user to input and use it in a function
function inputFif_widget {
  local REPLY
  autoload -Uz read-from-minibuffer

  # Create a sub-prompt, pre-populated with the current contents of the command line
  read-from-minibuffer 'Keywords to search: ' $LBUFFER $RBUFFER

  # Use the modified input to search and update comand line with it
  LBUFFER=$(echo "$(fif $REPLY)")
  RBUFFER=''

  # Put some additional text bellow command line
  # zle -M "Equivalent command: fif '$REPLY'"
}
zle -N inputFif_widget
bindkey '^X' 'inputFif_widget' # bind Ctrl+X to search