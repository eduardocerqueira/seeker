#date: 2023-01-20T17:03:24Z
#url: https://api.github.com/gists/514bedbe09c5d1ff3ed5f06686d80461
#owner: https://api.github.com/users/zhl146

#!/bin/bash
echo "Container Started"
export PYTHONUNBUFFERED=1
export PM2_HOME=/workspace/pm2

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

mkdir /root/.bittensor/wallets && cp -r /workspace/wallets_backup/* /root/.bittensor/wallets && cp /workspace/.bashrc  /root/.bashrc  && cp /workspace/.bash_history  /root/.bash_history
pm2 resurrect
sleep infinity