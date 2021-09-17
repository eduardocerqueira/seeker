#date: 2021-09-17T17:11:13Z
#url: https://api.github.com/gists/85fab066718f88e57e86bb06d6dfa9f8
#owner: https://api.github.com/users/chrisedrego

sudo tee -a <<EOT >> /etc/systemd/system/vault.service
[Unit]
Description=vault service
Requires=network-online.target
After=network-online.target
ConditionFileNotEmpty=/etc/vault/config.json

[Service]
EnvironmentFile=/etc/vault/env
Environment=GOMAXPROCS=2
Restart=on-failure
ExecStart=/usr/bin/vault server -config=/etc/vault/config.json
StandardOutput=/var/log/vault/output.log
StandardError=/var/log/vault/error.log
LimitMEMLOCK=infinity
ExecReload=/bin/kill -HUP $MAINPID
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOT