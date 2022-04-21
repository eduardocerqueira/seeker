#date: 2022-04-21T17:10:55Z
#url: https://api.github.com/gists/5133022f982239afb31231fb6e79549f
#owner: https://api.github.com/users/thesobercoder

# Almost Maximum Center Two Thirds
defaults write com.knollsoft.Rectangle almostMaximizeHeight -float 1
defaults write com.knollsoft.Rectangle almostMaximizeWidth -float 0.66

# Extra centering command with custom size - ^⌘⇧C (CTRL+CMD+SHIFT+C)
defaults write com.knollsoft.Rectangle specified -dict-add keyCode -float 8 modifierFlags -float 1442059
defaults write com.knollsoft.Rectangle specifiedHeight -float 720
defaults write com.knollsoft.Rectangle specifiedWidth -float 1280