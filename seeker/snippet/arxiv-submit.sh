#date: 2022-05-25T17:10:57Z
#url: https://api.github.com/gists/346797a2d766f1299700efbefbeb9f0e
#owner: https://api.github.com/users/matiscke

# To submit just at the right time (deadline 14:00 EST), 
# 
# How to use:
#  - Make sure your clock is synced to https://arxiv.org/localtime
#  - Prepare arxiv submission up to last page, move cursor over submission button
#  - execute below function which will produce a left-button mouse click at 14:00
#  - if too early, unsubmit, fix clock, try again the next day
# 
# Hopefully one day arxiv will randomize their submission ordering.

sleep $(($(date +%s -d '14:00 EST') - $(date +%s))) && xdotool click 1
