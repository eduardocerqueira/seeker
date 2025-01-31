#date: 2025-01-31T16:56:37Z
#url: https://api.github.com/gists/3a937dcbb1a1cdb0800472ff6d11ce4c
#owner: https://api.github.com/users/SavinRazvan

# Remove __pycache__ directories recursively
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove Windows Zone.Identifier metadata files
find . -type f -name "*:Zone.Identifier" -exec rm -f {} +

# Install pip-tools for dependency management
pip install pip-tools

# Generate a requirements.in file with all installed packages
pip freeze | cut -d= -f1 > requirements.in

# List installed packages in a formatted way
pip list --format=columns

# Compile dependencies into a locked requirements.txt with hashes
pip-compile --generate-hashes --output-file=requirements.txt requirements.in

# Compile dependencies into a locked requirements.txt without hashes
pip-compile --output-file=requirements.txt requirements.in

# Upgrade dependencies safely within version ranges
pip-compile --upgrade --output-file=requirements.txt requirements.in

# Check for dependency conflicts
pip check
