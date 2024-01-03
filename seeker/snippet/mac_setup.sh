#date: 2024-01-03T16:47:04Z
#url: https://api.github.com/gists/b641c2dfcd3700f8de9a8b38e70423aa
#owner: https://api.github.com/users/suvhotta

mac_setup () {
    YELLOW='\e[0;33m'
    GREEN='\e[0;32m'
    # Checking and installing brew
    if command -v brew &>/dev/null; then
        echo "${GREEN} Homebrew is already present."
    else
        echo "${YELLOW} Installing Homebrew."
        curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh
        echo "${GREEN} Homebrew installed successfully."
    fi

    echo "${YELLOW} Installing Rectangle."
    brew install --cask rectangle
    echo "${GREEN} Installed Rectangle."

    echo "${YELLOW} Installing visual-studio-code."
    brew install --cask visual-studio-code
    echo "${GREEN} Installed visual-studio-code."

    echo "${YELLOW} Installing git."
    brew install --cask git
    echo "${GREEN} Installed git."

    echo "${YELLOW} Installing zsh."
    curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh
    echo "${GREEN} Installed zsh."
    
    echo "${YELLOW} Installing zsh syntax highlighting."
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
    echo "${GREEN} Installed zsh syntax highlighting."

    echo "${YELLOW} Installing zsh auto suggestions."
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
    echo "${GREEN} Installed zsh auto suggestions."

    echo "${YELLOW} Installing powerlevel 10k."
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
    echo "${GREEN} Installed powerlevel 10k."

    echo "${YELLOW} Installing dbeaver."
    brew install --cask dbeaver-community
    echo "${GREEN} Installed dbeaver."

    echo "${YELLOW} Installing tfenv."
    brew install tfenv
    echo "${GREEN} Installed tfenv."

    echo "${YELLOW} Installing fonts."
    brew tap homebrew/cask-fonts
    brew install --cask font-fira-code
    echo "${GREEN} Installed fonts."
    
    echo "${YELLOW} Installing docker."
    brew install --cask docker
    echo "${GREEN} Installed docker."
}