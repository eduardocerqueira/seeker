#date: 2024-05-30T16:47:23Z
#url: https://api.github.com/gists/ad4b4811a8f658011e67e395da7f2c14
#owner: https://api.github.com/users/rikkarth

# this command will create an alias at 'git sla' which will print a very simple and beautiful log to the terminal
git config --global alias.sla 'log --color --graph --pretty=format:"%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset" --abbrev-commit --branches'
