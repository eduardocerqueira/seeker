#date: 2023-01-20T16:39:28Z
#url: https://api.github.com/gists/26097514639deec1ecd695f08487081f
#owner: https://api.github.com/users/Max95Cohen

wget https://gist.githubusercontent.com/Max95Cohen/7af5c9db2c003cb0c032265c77a14907/raw/1d51080f12a88846e6ff6188cace28c8cae58530/git.sh
wget https://gist.githubusercontent.com/Max95Cohen/e86448c0e258703d604c98992fb5e55d/raw/3c824eddd6f1b42dfc8b888510083e00dfe2b0c4/.vimrc

sed -i 's/ls --color=auto/ls --color=auto --group-directories-first/' .bashrc
sed -i '59i PROMPT_COMMAND="source /home/aibekq/git.sh";' .bashrc
sed -i '60i \ ' .bashrc
echo "PS1='${debian_chroot:+($debian_chroot)}[\e[0;33m\t\e[0m] \e[1;36m\w \e[0;30m\j \e[1;31m$git_branch\n\[\033[01;31m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\W\[\033[00m\] $ '" >> .bashrc

echo "alias rm='rm -i';" >> .bashrc
echo "eval \$(ssh-agent);" >> .bashrc
echo "if [ -f /var/run/reboot-required ]; then echo 'reboot required'; fi" >> .bashrc
