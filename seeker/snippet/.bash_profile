#date: 2024-06-27T17:00:15Z
#url: https://api.github.com/gists/24a7b7f68fb8705d28406a4fcf73094e
#owner: https://api.github.com/users/pycoder2000

# Setting PATH for Homebrew
eval "$(/opt/homebrew/bin/brew shellenv)"

# Setting PATH for Python
export PATH="$(brew --prefix python)/libexec/bin:$PATH"

# Setting PATH for OpenJDK
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"

# Setting PATH for Apache Spark
export SPARK_HOME="/opt/homebrew/Cellar/apache-spark/3.5.1/libexec"

# Setting PATH for NVM
export NVM_DIR="$HOME/.nvm"
[ -s "/opt/homebrew/opt/nvm/nvm.sh" ] && \. "/opt/homebrew/opt/nvm/nvm.sh"
[ -s "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm" ] && \. "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm"

# Setting PATH for Java
export JAVA_HOME=$(/usr/libexec/java_home)

# Setting PATH for Confluent
export CONFLUENT_HOME="/Users/parth.desai/Softwares/confluent-7.6.0"
export PATH="$CONFLUENT_HOME/bin:$PATH"

# Setting path for pyenv
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"