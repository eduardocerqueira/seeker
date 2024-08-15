#date: 2024-08-15T16:33:49Z
#url: https://api.github.com/gists/0ac67882692f23fd74fdadefb65db7d9
#owner: https://api.github.com/users/samsour

git rebase -i <parent-commit-sha> -x "git commit --amend --reset-author -CHEAD"