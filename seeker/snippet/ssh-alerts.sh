#date: 2024-09-23T17:10:13Z
#url: https://api.github.com/gists/de338ee5e3f1893460c53acf2eeeff98
#owner: https://api.github.com/users/stilliard

# Save file to /etc/profile.d/ssh-alerts.sh

# email on ssh login
if [ -n "$SSH_CLIENT" ]; then
    TEXT="$(date): ssh login to ${USER}@$(hostname -f)"
    TEXT="$TEXT from $(echo $SSH_CLIENT | awk '{print $1}')"
    echo $TEXT | mail -s "ssh login" your@email.here
fi
