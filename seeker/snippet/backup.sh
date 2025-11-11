#date: 2025-11-11T16:53:35Z
#url: https://api.github.com/gists/734acfc74aef52c4ff339929d7569f05
#owner: https://api.github.com/users/todd-dsm

~/.config/rsync/backups 2>&1 | tee /tmp/backups.log
Running LIVE backup
+ '[' -f /Users/USER/.config/rsync/special-backups.conf ']'
+ echo 'Backing up special ~/Library configs'
Backing up special ~/Library configs
+ mkdir -p /Users/USER/.config/admin/backup
+ IFS=,
+ read -r program source_path
+ [[ -z # backup random files and follow the format; $HOME is assumed ]]
+ [[ # backup random files and follow the format; $HOME is assumed = \#* ]]
+ continue
+ IFS=,
+ read -r program source_path
+ [[ -z cursor ]]
+ [[ cursor = \#* ]]
+ source_file='/Users/USER/Library/Application Support/Cursor/User/settings.json'
+ filename=settings.json
+ '[' -f '/Users/USER/Library/Application Support/Cursor/User/settings.json' ']'
+ cp '/Users/USER/Library/Application Support/Cursor/User/settings.json' /Users/USER/.config/admin/backup/cursor-settings.json
+ echo '  cursor'
  cursor
+ IFS=,
+ read -r program source_path
+ [[ -z foo ]]
+ [[ foo = \#* ]]
+ source_file=/Users/USER/bar.baz
+ filename=bar.baz
+ '[' -f /Users/USER/bar.baz ']'
+ cp /Users/USER/bar.baz /Users/USER/.config/admin/backup/foo-bar.baz
+ echo '  foo'
  foo
+ IFS=,
+ read -r program source_path
+ '[' -n '' ']'
+ set +x
sending incremental file list
./
.zsh_history
.config/admin/backup/cursor-settings.json
.config/admin/backup/foo-bar.baz
.config/rsync/backups
Documents/obsidian/tech/tech/tooling/AI-Tools/MCP Support.md
code/apple/rsync-backups/sources/special-backups.conf

Number of files: 97,289 (reg: 84,146, dir: 13,026, link: 117)
Number of created files: 0
Number of deleted files: 0
Number of regular files transferred: 6
Total file size: 75.00G bytes
Total transferred file size: 124.88K bytes
Literal data: 124.88K bytes
Matched data: 0 bytes
File list size: 172.03K
File list generation time: 0.001 seconds
File list transfer time: 0.000 seconds
Total bytes sent: 3.76M
Total bytes received: 14.50K

sent 3.76M bytes  received 14.50K bytes  1.08M bytes/sec
total size is 75.00G  speedup is 19,892.33
Backup complete: Tue Nov 11 08:35:34 PST 2025