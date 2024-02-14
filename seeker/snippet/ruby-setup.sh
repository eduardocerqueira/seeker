#date: 2024-02-14T16:52:42Z
#url: https://api.github.com/gists/bd9b5c377ead8c74d6b69240cd7de7ef
#owner: https://api.github.com/users/manelatun

#!/bin/bash

# Install build dependencies.
sudo apt update &&
sudo apt install autoconf patch build-essential rustc libssl-dev libyaml-dev \
  libreadline6-dev zlib1g-dev libgmp-dev libncurses5-dev libffi-dev libgdbm6 \
  libgdbm-dev libdb-dev uuid-dev -y

# Install rbenv.
curl -fsSL https://github.com/rbenv/rbenv-installer/raw/HEAD/bin/rbenv-installer | bash
echo >> ~/.bashrc
echo 'eval "$($HOME/.rbenv/bin/rbenv init - bash)"' >> ~/.bashrc
eval "$($HOME/.rbenv/bin/rbenv init - bash)"

# Install latest Ruby & Rails.
RUBY_VERSION=$(rbenv install -l | grep -v - | tail -1)
rbenv install $RUBY_VERSION
rbenv global $RUBY_VERSION
gem update --system
gem install rails

echo 'Restart your shell to use Ruby.'
