#date: 2025-06-19T16:54:13Z
#url: https://api.github.com/gists/2e69652872dc17437651a5630182b430
#owner: https://api.github.com/users/prathamlahoti123

# uv version
uv --version

# generate basic template to build an app
uv init

# install packages
uv add flask requests

# install all packages from requirements.txt
uv add -r requirements.txt

# install a tool globally
uv tool install ruff mypy

# run a tool without installation
uv tool run ruff check

# shortcut for 'uv tool run' command
uvx

# run a tool without installation using uvx
uvx ruff check

# uninstall one or multiple package
uv remove flask

# uninstall one or multiple tools
uv tool uninstall ruff

# list of installed tools
uv tool list

# upgrade a specific tool
uv tool upgrade ruff

# upgrade all installed tools
uv tool upgrade --all

# install packages using uv pip
uv pip install flask

# tree of installed packages and their dependencies
uv tree

# list of installed packages using uv pip
uv pip list

# run a script
uv run main.py

# recreate a virtual environment based on uv.lock file
uv sync
