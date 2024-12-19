#date: 2024-12-19T16:50:03Z
#url: https://api.github.com/gists/5b4701b7475961712cb01d00b431250a
#owner: https://api.github.com/users/jymchng

import os, sys # Possible to import most packages; go to [Installed Packages] tab, type in the names 
# delimited by comma of the packages you want to install.
# Click on [Add Package(s)]
# Then import those packages, e.g. import attrs

import logging
from logging import getLogger

logger = getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', stream=sys.stdout)

# Click on the dropdown button and choose [Format], then click on the [Format] button
# to see how the `fibonacci` function definition is reformatted.

# Define a function that takes an integer n and returns the nth number in the Fibonacci
# sequence.
def fibonacci(n):
    """Compute the nth number in the Fibonacci sequence."""
    x = 1
    if n == 0: return 0
    elif n == 1: return 1
    else: return fibonacci(n - 1) + fibonacci(n - 2)

# Use a for loop to generate and print the first 10 numbers in the Fibonacci sequence.
for i in range(10):
    print(fibonacci(i))
    logger.info(f"`fibonacci({i})` = {fibonacci(i)}")

# Click on the dropdown button and choose [Test], then click on the [Test] button
# to see how you can use pytest on this playground.
def test_fibonacci():
    for i in range(10):
        assert fibonacci(i) == fibonacci(i)
