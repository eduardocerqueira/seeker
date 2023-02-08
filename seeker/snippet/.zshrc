#date: 2023-02-08T16:44:26Z
#url: https://api.github.com/gists/2a2e8a1ddbbf329d783a78bb1bd581b9
#owner: https://api.github.com/users/Samir-Sattarov

# I'll be doing another one for Linux, but this one will give you 
# a pop up notification and sound alert (using the built-in sounds for macOS)

# Requires https://github.com/caarlos0/timer to be installed

# Mac setup for pomo
alias work="timer 60m && terminal-notifier -message 'Pomodoro'\
        -title 'Work Timer is up! Take a Break 😊'\
        -appIcon '~/Pictures/pumpkin.png'\
        -sound Crystal"
        
alias rest="timer 10m && terminal-notifier -message 'Pomodoro'\
        -title 'Break is over! Get back to work 😬'\
        -appIcon '~/Pictures/pumpkin.png'\
        -sound Crystal"