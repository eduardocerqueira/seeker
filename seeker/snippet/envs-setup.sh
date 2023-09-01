#date: 2023-09-01T16:52:50Z
#url: https://api.github.com/gists/1dbce5d091781fa1888a2031bfcaad6a
#owner: https://api.github.com/users/JustMrMendez

#!/bin/bash

# Function to configure Git
configure_git() {
  read -p "Enter your first and last name: " name
  read -p "Enter your email: " email
  git config --global user.name "$name"
  git config --global user.email "$email"
}

# Function to install Learnpack
install_learnpack() {
  npm install @learnpack/learnpack -g
}

# Function to clone repo and enter it
clone_repo() {
  while true; do
    read -p "Enter GitHub repo (username/reponame): " repo_info
    repo_link="https://github.com/$repo_info.git"
    
    echo "Attempting to clone $repo_link"

    git clone "$repo_link"
    
    if [ $? -eq 0 ]; then
      break
    else
      echo "Failed to clone repository. Please try again."
    fi
  done
  
  repo_name=$(basename "$repo_info")
  cd "$repo_name" || { echo "Directory not found"; exit 1; }
}

# Function to start Learnpack
start_learnpack() {
  learnpack start
}

# Execute functions
configure_git
install_learnpack
clone_repo
start_learnpack

echo "Setup complete."
