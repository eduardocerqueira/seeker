#date: 2023-08-22T17:03:40Z
#url: https://api.github.com/gists/c75d38f118e42ad0c46b074627f7cd4d
#owner: https://api.github.com/users/nickboldt

  # squash this commit into the previous one, using non-interactive interactive rebase
  # https://stackoverflow.com/questions/12394166/how-do-i-run-git-rebase-interactive-in-non-interactive-manner

# merge commits but keep comments
  GIT_EDITOR="sed -i -e 's/BAD/BAD/g'" GIT_SEQUENCE_EDITOR="sed -i -re '2,\$s/^pick /s /'" git rebase -i

# merge commits and drop comment
  GIT_EDITOR="sed -i -e 's/BAD/BAD/g'" GIT_SEQUENCE_EDITOR="sed -i -re '2,\$s/^pick /f /'" git rebase -i
