#date: 2024-03-27T17:03:21Z
#url: https://api.github.com/gists/10020472e5e3ee1ed6a38f1b22d718c0
#owner: https://api.github.com/users/ifkas

nano ~/.bash_profile
if [ -r ~/.bashrc ]; then
   source ~/.bashrc
fi

nano ~/.bashrc
alias sail='bash vendor/bin/sail'

# Then run sail by
sail up

# Or run in detached mode
sail up -d