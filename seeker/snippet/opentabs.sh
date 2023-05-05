#date: 2023-05-05T16:37:20Z
#url: https://api.github.com/gists/ce7323849170c35065779b5b4bf78d83
#owner: https://api.github.com/users/ccahill1117

# shell script to open iTerm tabs
osascript &>/dev/null <<EOF
# applescript
# below is simple example for opening tabs
# in the array below, you can put whatever you need for however many tabs

# declare array
set myList to {"cd Desktop", "cd Library"}

# loop thru array to open those tabs
repeat with theItem in myList
  tell application "iTerm"
      tell current window
          create tab with default profile
      end tell
      tell current tab of current window
          set _new_session to last item of sessions
      end tell
      tell _new_session
          select
          write text theItem
          write text "pwd"  
      end tell
  end tell
end repeat
EOF
