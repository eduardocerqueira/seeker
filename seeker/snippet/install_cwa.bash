#date: 2024-01-23T16:52:46Z
#url: https://api.github.com/gists/9758fa4f8ec759d307be09eba44fdd14
#owner: https://api.github.com/users/Degamisu

# Install Git

type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

# Clone Repo

gh auth login
wait(3)
gh repo clone Degamisu/Console-Weather-App && cd Console-Weather-App

# Install Dependencies

pip install -r requirements.txt
pyinstaller --onefile main.py

# Cleanup

rm -r build

# Run

cd dist
./main