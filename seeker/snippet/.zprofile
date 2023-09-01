#date: 2023-09-01T16:52:03Z
#url: https://api.github.com/gists/73c89f164534e4fce3fe43e06c85587a
#owner: https://api.github.com/users/ksquarekumar

# BASE

# CC
export CC="/usr/bin/clang"
export CXX=$CC++
export PYTHON_CONFIGURE_OPTS='--enable-optimizations --with-lto'
export PYTHON_CFLAGS='-march=native -mtune=native'

# I/O
ulimit -n 200000
ulimit -u 2048

# CANONICAL PATH
export PATH=$HOME/bin:/usr/local/bin:/usr/local/sbin:$PATH

# CUDA
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# COMPUTE
OMP_NUM_THREADS="$(nproc)"

# JAX/XLA
XLA_PYTHON_CLIENT_PREALLOCATE=false

# PYENV & PYTHON
export PIP_DEFAULT_TIMEOUT=100
export PYENV_VERSION="mambaforge-22.9.0-3"
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ubuntu/.pyenv/versions/mambaforge-22.9.0-3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ubuntu/.pyenv/versions/mambaforge-22.9.0-3/etc/profile.d/conda.sh" ]; then
        . "/home/ubuntu/.pyenv/versions/mambaforge-22.9.0-3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ubuntu/.pyenv/versions/mambaforge-22.9.0-3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/home/ubuntu/.pyenv/versions/mambaforge-22.9.0-3/etc/profile.d/mamba.sh" ]; then
    . "/home/ubuntu/.pyenv/versions/mambaforge-22.9.0-3/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

# PDM
export PATH=/home/ubuntu/.local/bin:$PATH