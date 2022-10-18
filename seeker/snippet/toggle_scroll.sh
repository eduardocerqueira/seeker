#date: 2022-10-18T17:32:20Z
#url: https://api.github.com/gists/36d5e4a09259983f700e9b2a29ab30c9
#owner: https://api.github.com/users/deepakmahakale

#!/bin/bash
osascript <<EOD
  tell application "System Preferences"
    (run)
    set current pane to pane "com.apple.preference.trackpad"
  end tell
  tell application "System Events"
    tell process "System Preferences"
      delay 0.6
      click radio button "Scroll & Zoom" of tab group 1 of window "Trackpad"
      click checkbox 1 of tab group 1 of window "Trackpad"
    end tell

    tell application "System Preferences" to quit
  end tell
EOD
