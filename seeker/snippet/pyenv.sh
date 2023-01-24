#date: 2023-01-24T16:41:45Z
#url: https://api.github.com/gists/1bf67e111b10f652b1770f53d7d8ea81
#owner: https://api.github.com/users/rodneyxr

# https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Install and switch to latest 3.10 version
pyenv install 3.10
pyenv global 3.10