#date: 2022-09-22T17:25:15Z
#url: https://api.github.com/gists/a4a937fa6f7b6660a6368ddfe36c72f0
#owner: https://api.github.com/users/mmarjani

cat <<'EOF' > /etc/systemd/system/ssht.service
[Unit]
Description=Setup a dynamic tunnel
After=network.target

[Service]
ExecStart=/usr/bin/ssh -o ServerAliveInterval=60  -o ClientAliveInterval=60 -o ExitOnForwardFailure=yes -nNT -D ${LOCAL_ADDR IP:PORT} ${REMOTE_USER}@${REMOTE_ADDRESS}
RestartSec=15
Restart=always
KillMode=mixed

[Install]
WantedBy=multi-user.target
EOF