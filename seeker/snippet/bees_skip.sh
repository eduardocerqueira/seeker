#date: 2024-06-28T16:27:48Z
#url: https://api.github.com/gists/526e8d958493b2bb45f8593b09472e0a
#owner: https://api.github.com/users/Infernio

# Put the UUID of the btrfs filesystem on which bees is stuck here.
sudo systemctl stop beesd@PUT_UUID_HERE.service
# Use the right BEESHOME here.
vim $BEESHOME/beescrawl.dat
# In Vim, run the following command:
# :%s/min_transid \(\d\+\) max_transid \(\d\+\)/min_transid \2 max_transid \2/g
# Same as before, put the right UUID here.
sudo systemctl start beesd@PUT_UUID_HERE.service
# If everything worked correctly, bees should start and drop down to 0% CPU usage after a few seconds.