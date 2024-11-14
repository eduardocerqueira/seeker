#date: 2024-11-14T16:54:06Z
#url: https://api.github.com/gists/478a75e647fcea242e424e4e694284dd
#owner: https://api.github.com/users/joewoz

from progress_bar import ProgressBar
from time import sleep

my_pg = ProgressBar(10)
for i in range(10):
    sleep(1)
    my_pg.update(i)
