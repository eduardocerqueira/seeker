#date: 2023-05-12T16:57:34Z
#url: https://api.github.com/gists/fc9476632798e35c768ae1024baaeaed
#owner: https://api.github.com/users/destag

#!/bin/bash

sudo apt install curl git inotify-tools

git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.11.3
curl -L https://fly.io/install.sh | sh

cat >> ~/.bashrc << 'EOF'

export FLYCTL_INSTALL="/home/gitpod/.fly"
export PATH="$FLYCTL_INSTALL/bin:$PATH"
. "$HOME/.asdf/asdf.sh"
EOF

. ~/.bashrc

# for local database
docker run --restart always --name postgres -e POSTGRES_PASSWORD=postgres -d -p 5432: "**********"

asdf plugin add erlang https://github.com/asdf-vm/asdf-erlang.git
asdf install erlang 25.3.2

asdf plugin-add elixir https://github.com/asdf-vm/asdf-elixir.git
asdf install elixir 1.14.4-otp-25

asdf global erlang 25.3.2
asdf global elixir 1.14.4-otp-25

mix local.hex
mix local.rebar --force

fly launch