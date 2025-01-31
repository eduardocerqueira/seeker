#date: 2025-01-31T17:05:41Z
#url: https://api.github.com/gists/69008439198a2a387e7fb4e04b0f4127
#owner: https://api.github.com/users/smeech

# Routines to follow the Espanso log for problems, without having to 
# type `espanso log` every time something goes wrong.
# Filters out routine messages.

# Linux
tail -F ~/.cache/espanso/espanso.log | grep -v "INFO"

# Windows
Get-Content "$env:LOCALAPPDATA\espanso\espanso.log" -Wait | Where-Object { $_ -notmatch "INFO" }
