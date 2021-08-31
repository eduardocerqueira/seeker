#date: 2021-08-31T01:09:02Z
#url: https://api.github.com/gists/556d8c8b95730871b472c4df1dd1f4c5
#owner: https://api.github.com/users/bberak

#!/bin/sh

# Source: https://stackoverflow.com/a/64836177/138392
# Prequisites: MacOS and Android Studio 4+ installed
# Usage: bash remove-android-studio.sh

# Deletes the Android Studio application
# Note that this may be different depending on what you named the application as, or whether you downloaded the preview version
rm -Rf /Applications/Android\ Studio.app

# Delete All Android Studio related preferences
# The asterisk here should target all folders/files beginning with the string before it
rm -Rf ~/Library/Preferences/Google/AndroidStudio*

# Deletes the Android Studio's plist file
rm -Rf ~/Library/Preferences/com.google.android.*

# Deletes the Android Emulator's plist file
rm -Rf ~/Library/Preferences/com.android.*

# Deletes mainly plugins (or at least according to what mine (Edric) contains)
rm -Rf ~/Library/Application\ Support/Google/AndroidStudio*

# Deletes all logs that Android Studio outputs
rm -Rf ~/Library/Logs/Google/AndroidStudio*

# Deletes Android Studio's caches
rm -Rf ~/Library/Caches/Google/AndroidStudio*

# Deletes older versions of Android Studio
rm -Rf ~/.AndroidStudio*