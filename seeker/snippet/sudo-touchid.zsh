#date: 2022-07-14T16:50:40Z
#url: https://api.github.com/gists/a3e518e3af7a94432ad6f6fef464f08b
#owner: https://api.github.com/users/git-commit

# Enable sudo via touch ID
grep -qxF "auth       sufficient     pam_tid.so" /etc/pam.d/sudo || sudo sed -i '' '1 a\
auth       sufficient     pam_tid.so
' /etc/pam.d/sudo