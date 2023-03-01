#date: 2023-03-01T16:49:07Z
#url: https://api.github.com/gists/5cb36fe4f6743d00f6ee1b338ddce4e0
#owner: https://api.github.com/users/davask

Get-ChildItem -Path "C:\Path\To\Dir" -Recurse | Select @{Name="MB Size";Expression={ "{0:N1}" -f ($_.Length / 1MB) }}, Fullname, LastWriteTime | Export-CSV -Path "C:\Path\To\Dir\list.csv" -NoTypeInformation