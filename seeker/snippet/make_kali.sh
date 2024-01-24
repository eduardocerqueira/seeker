#date: 2024-01-24T17:09:53Z
#url: https://api.github.com/gists/369f3c472c0503bac6087e4c3c9c8bde
#owner: https://api.github.com/users/Yair-Men


#!/usr/bin/bash


#### COLORS and Logger ####
color_Off='\033[0m' 
color_yellow='\033[0;33m'
color_red='\033[0;31m' 

function log_verbose() {
  echo -e "${color_yellow}$@${color_off}"
}

function log_err() {
  echo -e "${color_red}$@${color_off}"
}

#### Check root ####
if [ "$UID" -ne 0 ]
  then log_err "[-] Please run as root"
  exit
fi

cd /root

#### Update everything ####
log_verbose "Doing apt stuff"
apt-get update -y && apt-get full-upgrade -y && apt-get -y remove -y && apt-get autoclean -y

#### Change crapy zsh to bash ####
log_verbose "Changing Shell to bash"
chsh -s /bin/bash

#### Configure tmux ####
log_verbose "Making .tmux.conf"
cat << EOF > ~/.tmux.conf 
set -g history-limit 10000                # Buffer space
set -g allow-rename off                   # Disallow tmux to rename window
set-window-option -g mode-keys vi         # Changing emac to vi
set -g default-terminal "xterm-256color"  # Preserve the TERM env var
EOF

#### Install vscode ####
log_verbose "Installing vscode"
apt-get install wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list
rm -f packages.microsoft.gpg
apt-get install -y apt-transport-https
apt-get update -y
apt-get -y install code

#### Install chrome ####
log_verbose "Installing Chrome"
wget https://dl-ssl.google.com/linux/linux_signing_key.pub -O /tmp/google.pub
gpg --no-default-keyring --keyring /etc/apt/keyrings/google-chrome.gpg --import /tmp/google.pub
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
apt-get update -y
apt-get install -y google-chrome-stable

#### Cloning git repos, create venvs and install requirements ####
log_verbose "git cloning offensive git repos. Going to /opt"
cd /opt

# Clone each repo and save only the tool name to create venv with this name
git_repos=("ly4k/Certipy" "fortra/impacket" "p0dalirius/Coercer" "dirkjanm/krbrelayx" "dirkjanm/PKINITtools")
git_tool_name=()

for repo in ${git_repos[@]}; do
  git_tool_name+=(`echo -n $repo | cut -d/ -f2`)
  log_verbose "cloning $repo"
  git clone --depth 1 "https://github.com/$repo.git"
done

# Create venvs for all cloned repos"
echo "[!] Installing and creating venvs for all tolls at $HOME/venvs"
cd $HOME
python_path="`realpath -P $(which python)`"
mkdir $HOME/venvs

for repo in ${git_tool_name[@]}
do
  log_verbose "Creating venv for $repo"
  ${python_path} -m venv "venvs/$repo-venv"
done

# Install requirements.txt or setup.py for every repo in the correspond venv
log_verbose "Installing requirements or setup.py for each repoistory in the corresponding venv"
for repo in ${git_tool_name[@]}; do
  echo "[!] Sourcing venv and installing requirements for $repo"

  source "$HOME/venvs/$repo-venv/bin/activate"
  cd "/opt/$repo"

  # pip here no full path so we use the venv python
  if [ -f "requirements.txt" ] || [ -f "setup.py" ]
  then
      if [ -f "requirements.txt" ]; then
        pip install -r "./requirements.txt"
      fi

      if [ -f "setup.py" ]; then
        echo "[!] Found setup for $repo"
        pip install -e .
      fi

    else
      log_err "Didn't found requirements.txt or setup.py for $rep. skipping..."
    fi

  deactivate
done

#### Settings icon for root user with --no-sdanbox args on vscode and chrome ####
if [ ! -d "$HOME/.local/share/applications/" ]; then
  mkdir -p "$HOME/.local/share/applications/"
fi

cat << EOF > ~/.local/share/applications/code.desktop
[Desktop Entry]
Name=Visual Studio Code (root)
Comment=Code Editing. Redefined.
GenericName=Text Editor
Exec=/usr/share/code/code --no-sandbox --user-data-dir="~/.vscode" --unity-launch %F
Icon=vscode
Type=Application
StartupNotify=false
StartupWMClass=Code
Categories=TextEditor;Development;IDE;
MimeType=text/plain;inode/directory;application/x-code-workspace;
Actions=new-empty-window;
Keywords=vscode;

[Desktop Action new-empty-window]
Name=New Empty Window
Exec=/usr/share/code/code --no-sandbox --user-data-dir="~/.vscode" --new-window %F
Icon=vscode
EOF


# ToDo:
# - Add function to check status code of 0
# - Install sublime
# - Make icons for chrome that will run as root
# - Disable power settings
# - Create aliases for the root user for launching vscode and chrome with --no-sandbox
  # alias code='/usr/share/code/code --no-sandbox --user-data-dir="~/.vscode"'
  # alias chrome=/usr/bin/google-chrome-stable --no-sandbox %U
# - apt install xclip

