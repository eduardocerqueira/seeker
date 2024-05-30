#date: 2024-05-30T17:03:55Z
#url: https://api.github.com/gists/44863e97f34ea41b4bb57b84d4d32649
#owner: https://api.github.com/users/rikkarth

# this command will configure your dual monitors for horizontal + vertical setup, 
# where horizontal is primary and vertical is secondary (right)
# you can find the name of each monitor by typing 'xrandr'

xrandr --output $PRIMARY_MON --primary --mode 1920x1080 --rotate normal --output $SECONDARY_MON --mode 1920x1080 --rotate right --pos 1920x-300

# additionally the command bellow can be placed in .config/i3/config

exec --no-startup-id xrandr --output $PRIMARY_MON --primary --mode 1920x1080 --rotate normal --output $SECONDARY_MON --mode 1920x1080 --rotate right --pos 1920x-300
