#date: 2025-02-06T16:51:36Z
#url: https://api.github.com/gists/9f7f7b5c662dbcbcc161912894935d37
#owner: https://api.github.com/users/husamahmud

# Step 1: Open .zshrc in the Terminal
vi ~/.zshrc
# vi ~/.bashrc  # For Bash

# Step 2: Add the Alias
push() {
  git add .
  git commit -m "$1"
  git push
}

# Step 3: Reload .zshrc to apply changes
source ~/.zshrc
# source ~/.bashrc  # For Bash

# Run the `push` command with your commit message
push "my commit message"