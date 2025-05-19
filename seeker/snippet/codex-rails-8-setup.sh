#date: 2025-05-19T16:50:04Z
#url: https://api.github.com/gists/74e1c6d0ed01011ebeffc11797d19198
#owner: https://api.github.com/users/FrancescoK

#!/usr/bin/env bash
set -euo pipefail

apt-get update -qq
apt-get install -yqq build-essential curl git libssl-dev libreadline-dev zlib1g-dev libyaml-dev libvips-dev libheif-dev libde265-dev xz-utils ca-certificates wget libpq-dev vim unzip gnupg2 dirmngr

RUBY_VERSION=3.4.4

RUBY_ROOT=/opt/rubies/ruby-${RUBY_VERSION}

mkdir -p "$RUBY_ROOT"

curl -sSL \
  "https://rubies.travis-ci.org/ubuntu/24.04/x86_64/ruby-${RUBY_VERSION}.tar.bz2" \
  -o ruby-${RUBY_VERSION}.tar.bz2
tar -xjf ruby-${RUBY_VERSION}.tar.bz2 -C "$RUBY_ROOT" --strip-components=1
rm ruby-${RUBY_VERSION}.tar.bz2

echo "export PATH=${RUBY_ROOT}/bin:\$PATH" >/etc/profile.d/99-ruby.sh
export PATH=${RUBY_ROOT}/bin:$PATH

# Uncomment if you run Postgres. Make sure to add postgres to apt-get install on line 5
# pg_ctlcluster --skip-systemctl-redirect 16 main start
# su - postgres -c "createuser -s $(whoami)"

gem update --system --no-document
gem install bundler --no-document
npm install

# Using Playwright for system tests instead of Selenium
npx playwright install chromium

bundle

bin/rails db:prepare