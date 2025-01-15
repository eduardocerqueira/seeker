#date: 2025-01-15T17:12:25Z
#url: https://api.github.com/gists/4f95f095ad1704fbb3728aa9b16d6913
#owner: https://api.github.com/users/dosbenjamin

#!/bin/sh

chsh -s $(which zsh)
curl -o ~/.zshrc https://gist.github.com/dosbenjamin/61b86c4d07e7c5b5fbd139f31dae0068

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
