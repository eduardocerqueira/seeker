#date: 2025-03-26T16:57:59Z
#url: https://api.github.com/gists/b51693d71072acd1b215649387faf70e
#owner: https://api.github.com/users/albertrodriguezdev

dconf dump /org/gnome/shell/extensions/dash-to-dock/ | awk -F '=' '
  BEGIN {print "---"}
  /=/ {
    gsub(/^ +| +$/, "", $1);
    gsub(/^ +| +$/, "", $2);
    print "- key: \"" $1 "\"";
    print "  value: \"" $2 "\"";
  }
' > dash-to-dock.yaml