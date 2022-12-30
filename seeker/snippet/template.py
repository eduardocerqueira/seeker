#date: 2022-12-30T16:35:28Z
#url: https://api.github.com/gists/087ba93b1ff2e67bd2bbc887ff641187
#owner: https://api.github.com/users/fivesevendev

#


import timeit
import time


def numFind():
    pass


if __name__ == '__main__':
    startTime = timeit.default_timer()
    print(">>>>>", time.asctime(), "<<<<<\n")
    print("Result:", numFind())
    print("Run Time Was {:.4F} Seconds".format(timeit.default_timer() - startTime))
    print("\n>>>>>", time.asctime(), "<<<<<")