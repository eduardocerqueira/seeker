#date: 2024-09-09T17:04:54Z
#url: https://api.github.com/gists/0c21cd99ac05595bbf68577865023534
#owner: https://api.github.com/users/manesec

#!/usr/bin/expect

if { $argc != 1 } {
    puts "Usage: $argv0 <config.ovpn>"
    exit 1
}

set configFile [lindex $argv 0]

set timeout -1

spawn openvpn --config $configFile
expect "Enter Auth Username:"
send -- "<username>\r"
expect  "**********"Enter Auth Password: "**********"
send -- "<password>\r"
interact
