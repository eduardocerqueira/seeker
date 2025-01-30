#date: 2025-01-30T16:53:38Z
#url: https://api.github.com/gists/18b23955d3785c5b9e3754fc3dd37f49
#owner: https://api.github.com/users/micheldegeofroy

#!/bin/sh

if [ `whoami` != "root" ]; then
  echo "Run as sudo to update your motd."
  exit 1
fi

MOTD_DIR=/etc/update-motd.d/

mkdir -p "${MOTD_DIR}"
wget -qN 'https://gist.githubusercontent.com/meeDamian/0006c766340e0afd16936b13a0c7dbd8/raw/8b9a821feccd3c95d62e7cc937146db469fadc8f/10-uname' -P "${MOTD_DIR}"
wget -qN 'https://gist.githubusercontent.com/meeDamian/0006c766340e0afd16936b13a0c7dbd8/raw/8b9a821feccd3c95d62e7cc937146db469fadc8f/20-raspberry-bitcoin' -P "${MOTD_DIR}"
wget -qN 'https://gist.githubusercontent.com/meeDamian/0006c766340e0afd16936b13a0c7dbd8/raw/3552b9a418d81d12ae004850cbcb4578a43fdfca/23-raspberry-lnd' -P "${MOTD_DIR}"
wget -qN 'https://gist.githubusercontent.com/meeDamian/0006c766340e0afd16936b13a0c7dbd8/raw/3552b9a418d81d12ae004850cbcb4578a43fdfca/26-raspberry-lightning' -P "${MOTD_DIR}"
wget -qN 'https://gist.githubusercontent.com/meeDamian/0006c766340e0afd16936b13a0c7dbd8/raw/8b9a821feccd3c95d62e7cc937146db469fadc8f/30-swap-warning' -P ${MOTD_DIR}

chmod +x "${MOTD_DIR}10-uname" "${MOTD_DIR}20-raspberry-bitcoin" "${MOTD_DIR}23-raspberry-lnd" "${MOTD_DIR}26-raspberry-lightning" "${MOTD_DIR}30-swap-warning"

run-parts --lsbsysinit /etc/update-motd.d