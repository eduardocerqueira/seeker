#date: 2024-08-20T16:44:38Z
#url: https://api.github.com/gists/97b994f6766f4fe6b66e84c97104ebe5
#owner: https://api.github.com/users/tuliocasagrande

#!/bin/bash

# -e  Exit immediately if a command exits with a non-zero status.
# -u  Treat unset variables as an error when substituting.
# -x  Print commands and their arguments as they are executed.
set -eux

# Install extensions and depending libraries
pip install -U \
    jupyterlab-spellchecker \
    jupyterlab-code-formatter \
    jupyterlab-notifications \
    black flake8 isort

# Configurations
# I prefer flake8 over pylint and the others. Also, setting max line to 100 and ignoring a few errors
CONFIG_DIR=.jupyter/lab/user-settings/@jupyter-lsp/jupyterlab-lsp
CONFIG_FILE=plugin.jupyterlab-settings
CONFIG_PATH="$CONFIG_DIR/$CONFIG_FILE"
if test -f $CONFIG_PATH; then
    echo "$CONFIG_PATH already exists. Skipping..."
else
    echo "Creating $CONFIG_PATH"
    mkdir -p $CONFIG_DIR
    cat >$CONFIG_PATH <<EOF
{
  "language_servers": {
    "pylsp": {
      "serverSettings": {
        "pylsp.plugins.flake8.enabled": true,
        "pylsp.plugins.flake8.ignore": ["E303", "E402"],
        "pylsp.plugins.flake8.maxLineLength": 100,
        "pylsp.plugins.pycodestyle.enabled": false,
        "pylsp.plugins.pyflakes.enabled": false,
        "pylsp.plugins.pylint.enabled": false
      }
    }
  }
}
EOF
fi

# Format on save and other customizations
CONFIG_DIR=.jupyter/lab/user-settings/jupyterlab_code_formatter
CONFIG_FILE=settings.jupyterlab-settings
CONFIG_PATH="$CONFIG_DIR/$CONFIG_FILE"
if test -f $CONFIG_PATH; then
    echo "$CONFIG_PATH already exists. Skipping..."
else
    echo "Creating $CONFIG_PATH"
    mkdir -p $CONFIG_DIR
    cat >$CONFIG_PATH <<EOF
{
  "preferences": {
    "default_formatter": {
      "python": ["isort", "black"]
    }
  },
  "black": {
    "line_length": 100,
    "string_normalization": true,
    "magic_trailing_comma": true
  },
  "isort": {
    "line_length": 100
  },
  "formatOnSave": true
}
EOF
fi

# Some customizations for Jupyter Notebook
CONFIG_DIR=.jupyter/lab/user-settings/@jupyterlab/notebook-extension
CONFIG_FILE=tracker.jupyterlab-settings
CONFIG_PATH="$CONFIG_DIR/$CONFIG_FILE"
if test -f $CONFIG_PATH; then
    echo "$CONFIG_PATH already exists. Skipping..."
else
    echo "Creating $CONFIG_PATH"
    mkdir -p $CONFIG_DIR
    cat >$CONFIG_PATH <<EOF
{
  "codeCellConfig": {
    "autoClosingBrackets": true,
    "highlightTrailingWhitespace": true,
    "lineNumbers": false,
    "lineWrap": false
  },
  "markdownCellConfig": {
    "autoClosingBrackets": true,
    "highlightTrailingWhitespace": true,
    "lineNumbers": false,
    "matchBrackets": true
  }
}
EOF
fi

# Customize web-browser notifications
CONFIG_DIR=.jupyter/lab/user-settings/jupyterlab-notifications
CONFIG_FILE=plugin.jupyterlab-settings
CONFIG_PATH="$CONFIG_DIR/$CONFIG_FILE"
if test -f $CONFIG_PATH; then
    echo "$CONFIG_PATH already exists. Skipping..."
else
    echo "Creating $CONFIG_PATH"
    mkdir -p $CONFIG_DIR
    cat >$CONFIG_PATH <<EOF
{
  "cell_number_type": "cell_execution_count",
  "last_cell_only": true,
}
EOF
fi

# Once components are installed and configured, restart Jupyter to make sure everything propagates
echo "Restarting Jupyter server..."
restart-jupyter-server
