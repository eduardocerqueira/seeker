#date: 2023-04-24T17:00:10Z
#url: https://api.github.com/gists/798b611a0e9d2f889739e370de6306ac
#owner: https://api.github.com/users/HarshScindia

# 1) ssh-keygen 
# give file name and passcode
ssh-keygen

# 2) eval `ssh-agent`
eval `ssh-agent

# 3) ssh-add ~/.ssh/<private_key_file>  
# private_key_file name of the file given in step 1
ssh-add ~/.ssh/<private_key_file>  

# 4) cat ~/.ssh/<private_key_file>.pub
# Add key to your github account
cat ~/.ssh/<private_key_file>.pub

# 5)
ssh -T git@github.com