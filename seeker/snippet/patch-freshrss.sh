#date: 2024-12-19T17:06:01Z
#url: https://api.github.com/gists/91e699ee8aacaabd308c41f0b09f4ab6
#owner: https://api.github.com/users/alecmaly


apt update
apt install -y git
git config --global user.email "test@test.com"


# Step 1: Clone the repository to a temporary directory.
git clone https://github.com/alecmaly/FreshRSS.git temp-repo

# Step 2: Checkout the specific commit in the temporary repository.
cd temp-repo
git checkout sort_by_publish

# sync with freshrss
git remote add upstream https://github.com/FreshRSS/FreshRSS.git
git fetch upstream
git merge upstream/edge
git commit -m "merged"


# Step 3: Copy all files to your current directory, overwriting existing files.
cd ..
cp -r temp-repo/* ./  # Recursive copy of all files from the temporary directory.

# Step 4: Remove the temporary repository as it's no longer needed.
rm -rf temp-repo
