#date: 2024-06-18T17:00:23Z
#url: https://api.github.com/gists/e1b928d5f7c90c5aee063f350227a1c4
#owner: https://api.github.com/users/gitmpr

#!/usr/bin/env bash

# Define the menu options
menu_options=("Option 0" "Option 1" "Option 2" "Option 3" "Option 4" "Option 5")
num_menu_options=${#menu_options[@]}

# enable/disable looping of the selection
looping=0


# Initialize the selected option index
selected=0

# Initialize array to track selected options
selected_options=()

# Function to print the menu
print_menu() {
  echo "Press [space] or [tab] to toggle an entry and then press [enter] to accept the total selection"
  for i in "${!menu_options[@]}"; do
    if [[ " ${selected_options[@]} " =~ " $i " ]]; then
      checkbox="[*]"
    else
      checkbox="[ ]"
    fi

    if [ $i -eq $selected ]; then
      echo -e "\e[7m> $checkbox ${menu_options[$i]}\e[m"
    else
      echo "  $checkbox ${menu_options[$i]}"
    fi
  done
}

# Move the cursor back up
reset_cursor() {
  echo -en "\e["$(($num_menu_options + 1))"A"
}

# Function to handle keypress events
keypress() {
  local key
  IFS= read -rsn1 key
  if [[ $key == $'\x1b' ]]; then
    read -rsn2 key
  fi

  case "$key" in
    '[A'|'k') # Up arrow
      ((selected--))
      if [ $selected -lt 0 ]; then

                #selected=$((${#menu_options[@]} - 1))
                #selected=0

                if [ "$looping" -eq 1 ]; then
                  selected=$((${#menu_options[@]} - 1))
                else
                  selected=0
                fi

      fi
      ;;
    '[B'|'j') # Down arrow
      ((selected++))
      if [ $selected -ge ${#menu_options[@]} ]; then

                #selected=0
                #selected=$((${#menu_options[@]} - 1))

                if [ "$looping" -eq 1 ]; then
                  selected=0
                else
                  selected=$((${#menu_options[@]} - 1))
                fi
      fi
      ;;
    $'\t'| ' ') # Tab or Space key
      if [[ " ${selected_options[@]} " =~ " $selected " ]]; then
        # If already selected, deselect it
        selected_options=("${selected_options[@]/$selected}")
      else
        # If not selected, select it
        selected_options+=($selected)
      fi
      ;;
    '') # Enter key
      echo "selected_options =" ${selected_options[@]}
      echo "Selected options:"
      # Use a loop to iterate through indices to print selected options
      for i in "${!menu_options[@]}"; do
        if [[ " ${selected_options[@]} " =~ " $i " ]]; then
          echo "${menu_options[$i]}"
        fi
      done
      exit 0
      ;;
  esac
}

showcursor() {
  echo -en "\033[?25h"
}

hidecursor() {
  echo -en "\033[?25l"
}

# Trap the EXIT signal to ensure cursor is restored before exiting
trap showcursor EXIT

hidecursor

print_menu
while true; do
  reset_cursor
  print_menu
  keypress
done

