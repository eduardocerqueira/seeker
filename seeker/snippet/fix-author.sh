#date: 2023-02-22T16:47:51Z
#url: https://api.github.com/gists/521953d973e9b23aa3dd92f21725e5a1
#owner: https://api.github.com/users/ahodieb

# Reset the author for multiple commits while presevering the original commit date
git -c rebase.instructionFormat='%s%nexec GIT_COMMITTER_DATE="%cD" GIT_AUTHOR_DATE="%aD" git commit --amend --no-edit --reset-author' rebase -r HEAD~2