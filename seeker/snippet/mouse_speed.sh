#date: 2023-08-09T17:05:28Z
#url: https://api.github.com/gists/ae36a8656efb5ea5a862b74e4592cf75
#owner: https://api.github.com/users/sethjback

# Get the device ID
xinput list

# Changn the accelearation
input --set-prop 19 'libinput Accel Speed' 1

# Change the matrix
# this will have the most effect. The 2.4 numbers are the ones to change
# See https://unix.stackexchange.com/questions/90572/how-can-i-set-mouse-sensitivity-not-just-mouse-acceleration
xinput set-prop 19 "Coordinate Transformation Matrix" 2.4 0 0 0 2.4 0 0 0 1
