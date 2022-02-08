#date: 2022-02-08T17:00:34Z
#url: https://api.github.com/gists/d141eeb9a0033d5b7c40fc2aa03f4d49
#owner: https://api.github.com/users/0187773933

# !/usr/bin/env osascript
# https://pfiddlesoft.com/uibrowser/
on run {input} # Don't need the parameters arguments
	set thisFile to input
	set fileName to name of (info for thisFile)
	tell application "System Events"
		tell application "Typora"
			activate
			open thisFile
		end tell
		delay 1
		tell process "Typora"
			click menu item "PDF" of menu 1 of menu item "Export" of menu 1 of menu bar item "File" of menu bar 1
			# repeat until exists window "Save" of sheet 1
			# end repeat
			delay 1
			click button "Save" of sheet 1 of window fileName
			delay 1
			click button 1 of window fileName
		end tell
	end tell
end run