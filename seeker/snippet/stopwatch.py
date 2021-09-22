#date: 2021-09-22T17:07:31Z
#url: https://api.github.com/gists/cfd7979de8ab6f17f8fec144f0882a3d
#owner: https://api.github.com/users/RaphaelGoutmann

# stopwatch.py

import time 

'''

    (-.-) This code needs to be improved because (-.-)
                 there's lot of bugs                    

'''

# u_u
class Stopwatch:
    def __init__(self):
        self.startTime = 0

    def start(self):
        self.startTime = time.monotonic()

    def reset(self):
        self.startTime = 0

    def time(self):
        return (time.time() - self.startTime)
