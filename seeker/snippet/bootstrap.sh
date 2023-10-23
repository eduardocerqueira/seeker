#date: 2023-10-23T16:39:43Z
#url: https://api.github.com/gists/47eae442050ff9c4e63fd5400d2556f1
#owner: https://api.github.com/users/simloo

#!/bin/bash

if ! command -v brew --version &> /dev/null; then
    echo "Install homebrew"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    (echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> /Users/$USER/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
    export PATH="/opt/homebrew/bin:$PATH"
fi

if [[ ! -d '/Applications/Google Chrome.app' ]]; then
    echo "Install google chrome"
    brew install --cask google-chrome
    open -a "Google Chrome" --args --make-default-browser
fi

if [[ ! -d '/Applications/Visual Studio Code.app' ]]; then
    echo "Install visual studio code"
    brew install --cask visual-studio-code
fi

if [[ ! -d '/Applications/Slack.app' ]]; then
    echo "Install Slack"
    brew install --cask slack
fi

if [[ ! -d '/Applications/Notion.app' ]]; then
    echo "Install Notion"
    brew install --cask notion
fi

if [[ ! -d '/Applications/Docker.app' ]]; then
    echo "Install Docker"
    brew install --cask docker
fi

if [[ ! -d '/Applications/1Password.app' ]]; then
    echo "Install 1Password"
    brew install --cask 1Password
fi

curl -sL https://binaries.twingate.com/client/macos/2023.250.4843+sa/Twingate.app.zip -o /tmp/twingate.app.zip
unzip -o /tmp/twingate.app.zip -d /Applications