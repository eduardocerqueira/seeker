#date: 2023-07-24T17:00:34Z
#url: https://api.github.com/gists/f1df7e6b231409ba0b25cb864b7a443e
#owner: https://api.github.com/users/camillanapoles


# $ brew install pyenv #pay attention to caveats ($ brew info pyenv)
# $ brew install pyenv-virtualenv 


# this goes into .zshrc
export PYENV_ROOT=/usr/local/var/pyenv
if which pyenv > /dev/null; then eval "$(pyenv init -)"; fi
if which pyenv-virtualenv-init > /dev/null; then eval "$(pyenv virtualenv-init -)"; fi

# USE LIKE THIs  
# $ pyenv install miniconda3-latest
# $ pyenv global miniconda3-latest
# $ conda create -n my_conda_env requests

# $ pyenv versions
#  system
#* miniconda3-latest (set by /Users/pocin/.python-version)
#  miniconda3-latest/envs/my_conda_env

# to activate conda virtualenv do
# pyenv activate my_conda_env

# For activating conda env when entering directory
# $ cd /path/to/dir
# $ pyenv local my_conda_env
