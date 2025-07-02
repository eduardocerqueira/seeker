#date: 2025-07-02T16:58:46Z
#url: https://api.github.com/gists/9e238685430749c608a70f952b57fccc
#owner: https://api.github.com/users/Jatapiaro

echo "qwerty" | pbcopy
sleep 0.2
osascript -e 'tell application "System Events" to keystroke "v" using {command down}'
(sleep 10 && pbcopy < /dev/null) &