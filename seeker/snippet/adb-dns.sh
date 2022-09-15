#date: 2022-09-15T17:18:58Z
#url: https://api.github.com/gists/e01c48b77a73dbd21ea89f541941efcc
#owner: https://api.github.com/users/rwp0

# to disable private dns
adb shell settings put global private_dns_mode off

# to enable private dns with hostname (example with dns.adguard.com)
adb shell settings put global private_dns_mode hostname
adb shell settings put global private_dns_specifier dns.adguard.com