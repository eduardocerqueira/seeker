#date: 2022-01-26T17:11:42Z
#url: https://api.github.com/gists/1f05e6786e73f1e99604cc61d1fc09a3
#owner: https://api.github.com/users/ChristianGrimberg

# Add this lines al te end of the file \\wsl.localhost\{disribution}\home\{username}\.bashrc

# enable GPG signing
export GPG_TTY=$(tty)

if [ ! -f ~/.gnupg/S.gpg-agent ]; then
    eval $( gpg-agent --daemon --options ~/.gnupg/gpg-agent.conf &>/dev/null )
fi

export GPG_AGENT_INFO=${HOME}/.gnupg/S.gpg-agent:0:1
