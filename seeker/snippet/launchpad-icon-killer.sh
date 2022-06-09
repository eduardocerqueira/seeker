#date: 2022-06-09T17:16:33Z
#url: https://api.github.com/gists/47c83bf360a66daf8089130d6b815407
#owner: https://api.github.com/users/rexarski

# Substitute APPNAME with the exact icon/app name

sqlite3 $(find /private/var/folders -name com.apple.dock.launchpad 2>/dev/null)/db/db \
"DELETE FROM apps WHERE title='APPNAME';" && \
killall Dock
