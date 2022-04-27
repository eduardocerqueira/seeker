#date: 2022-04-27T17:11:38Z
#url: https://api.github.com/gists/72347dd63a7cac000f91ed105c629cb0
#owner: https://api.github.com/users/blakek

wait-until-online() {
  until curl --head https://ddg.gg; do sleep 10; done
}

# Example usage with _notify
wait-until-online && _notify 'Back online!'

# One-liner for vanilla macOS
until curl --head https://ddg.gg; do sleep 10; done; osascript -e 'display notification "Back online!"'
