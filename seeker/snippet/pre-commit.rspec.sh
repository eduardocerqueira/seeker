#date: 2021-10-06T16:55:51Z
#url: https://api.github.com/gists/cb69ee9f6bd2b4d68c5a1676d3124520
#owner: https://api.github.com/users/LuisOsnet

#!/usr/bin/env bash

# Can use this in an IDE it can't load rvm environment.
# and charge the rvm environment.
# When you use git only in shell commands, ruby script 'pre-commit.rspec.rb' is enough. You only need to move it to pre-commit.
# If you want use this, move it to pre-commit, otherwise move pre-commit.rspec.rb to pre-commit.

if [[ -s "$HOME/.rvm/scripts/rvm" ]] ; then

  # First try to load from a user install
  source "$HOME/.rvm/scripts/rvm"

elif [[ -s "/usr/local/rvm/scripts/rvm" ]] ; then

  # Then try to load from a root install
  source "/usr/local/rvm/scripts/rvm"

else

  printf "ERROR: An RVM installation was not found.\n"

fi

rvm reload
ruby .git/hooks/pre-commit.rspec.rb
