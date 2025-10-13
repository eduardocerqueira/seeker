#date: 2025-10-13T16:54:14Z
#url: https://api.github.com/gists/11b2e38339e77ad1b3854733cdd97bbe
#owner: https://api.github.com/users/Kamesuta

#!/bin/bash

# AppDataのパスを取得
# 参考: https://gist.githubusercontent.com/andsens/2ebd7b46c9712ac205267136dc677ac1/raw/574f8c96fc3961fa8f953ee9335a9de3388ba256/gpg-agent-relay
winhome=$(cmd.exe /c "<nul set /p=%UserProfile%" 2>/dev/null || true)
slashwinhome=$(wslpath -m $(wslpath -u $winhome)) # C:/Users/ユーザー名 の形式に変換

# WindowsのGPGを使う設定
# https://qiita.com/Ouvill/items/4d96c3e1fd955fd95cf8
wsl2_ssh_pageant_bin="$HOME/.ssh/wsl2-ssh-pageant.exe"
config_path="$slashwinhome/AppData/Local/gnupg"

if test -x "$wsl2_ssh_pageant_bin"; then
  # S.gpg-agent
  export GPG_AGENT_SOCK=$(gpgconf --list-dirs agent-socket)
  if ! ss -a | grep -q "$GPG_AGENT_SOCK"; then
    rm -rf "$GPG_AGENT_SOCK"
    (setsid nohup socat UNIX-LISTEN:"$GPG_AGENT_SOCK,fork" EXEC:"$wsl2_ssh_pageant_bin -gpgConfigBasepath \"$config_path\" -gpg S.gpg-agent" >/dev/null 2>&1 &)
  fi

  # S.gpg-agent.extra
  export GPG_AGENT_EXTRA_SOCK=$(gpgconf --list-dirs agent-extra-socket)
  if ! ss -a | grep -q "$GPG_AGENT_EXTRA_SOCK"; then
    rm -rf "$GPG_AGENT_EXTRA_SOCK"
    (setsid nohup socat UNIX-LISTEN:"$GPG_AGENT_EXTRA_SOCK,fork" EXEC:"$wsl2_ssh_pageant_bin -gpgConfigBasepath \"$config_path\" -gpg S.gpg-agent.extra" >/dev/null 2>&1 &)
  fi
else
  echo >&2 "WARNING: $wsl2_ssh_pageant_bin is not executable."
fi