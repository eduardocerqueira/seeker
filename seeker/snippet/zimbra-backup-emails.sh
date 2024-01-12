#date: 2024-01-12T16:59:36Z
#url: https://api.github.com/gists/ee2de08026b1bc953c3ca59ba4c939d6
#owner: https://api.github.com/users/0x00dec0de

# The command below creates a tgz file with all emails for user@domain.com in .eml format:
# execute as root
/opt/zimbra/bin/zmmailbox -z -m user@domain.com getRestURL "//?fmt=tgz" > /tmp/account.tgz

# You can do the same via a REST URL:
wget http://ZIMBRA.SERVER/home/user@domain.com/?fmt=tgz

# to restore email:
/opt/zimbra/bin/zmmailbox -z -m user@domain.com postRestURL "//?fmt=tgz&resolve=reset" /tmp/account.tgz

# The resolve= parameter has several options:
# - skip:    ignores duplicates of old items, it’s also the default conflict-resolution.
# - modify:  changes old items.
# - reset:   will delete the old subfolder (or entire mailbox if /).
# - replace: will delete and re-enter them.
