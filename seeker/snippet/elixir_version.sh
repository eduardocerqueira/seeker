#date: 2024-06-25T16:59:47Z
#url: https://api.github.com/gists/b71259c7aaddbd9068fbcda94e8c8cce
#owner: https://api.github.com/users/w2u2u

#!/bin/bash

# Check Erlang version
echo "Erlang version:"
erl -version 2>&1

# Check Elixir version
echo -e "\nElixir version:"
elixir --version

# Check Mix version
echo -e "\nMix version:"
mix --version

# Check Phoenix version
echo -e "\nPhoenix version:"
mix phx.new --version