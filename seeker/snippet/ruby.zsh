#date: 2026-01-19T17:07:48Z
#url: https://api.github.com/gists/8b04cca445c580933c906a779aebb90e
#owner: https://api.github.com/users/chip

# This file is located at ~/.zsh/ruby.zsh and sourced from ~/.zshrc

# chruby settings
source /opt/homebrew/opt/chruby/share/chruby/chruby.sh
source /opt/homebrew/opt/chruby/share/chruby/auto.sh
chruby ruby-3.4.8

# ruby env vars for installing docs
export RUBY_CONFIGURE_OPTS="--with-libyaml-dir=$(/opt/homebrew/bin/brew --prefix libyaml) --enable-install-doc"
export RI_BASE_DIR="$HOME/.local/share/rdoc"

# Ruby docs with color
riz() {
  [[ -z "$1" ]] && echo "riz needs a Ruby method or class as an argument (Array, Array#map, map)" && return 0

  ri "$1" \                             # pass argument to Ruby documation API cli
    | perl -pe 's/.?\x08//g' \          # remove backspace characters
    | sed "s/'\(s\) /'\\\\''\1 /g" \    # escape occurrences of 's (apostrophe s)
    | sed "s/'\(t\) /'\\\\''\1 /g" \    # escape occurrences of 't (apostrophe t)
    | bat -l ruby                       # pipe documentation to bat using ruby language highlighting
}
