#date: 2021-12-28T16:48:03Z
#url: https://api.github.com/gists/1a737adfce78e25db84467a971502989
#owner: https://api.github.com/users/Exception4U

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'degreeOfArray' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY arr as parameter.
#

def degreeOfArray(arr):
    # Write your code here
    frequncyCount = {}
    startRangeCount = {} ## keeping track of start and end coordinate
    endRangeCount = {}
    for i,j in zip(arr,range(len(arr))):
        try:
            frequncyCount[i] +=1
            endRangeCount[i]=j
        except:
            frequncyCount[i]=1
            startRangeCount[i]=j
            endRangeCount[i]=j   
    degree = max(frequncyCount.values()) 
    degreeCount = {}
    for i in startRangeCount:
        if frequncyCount[i]==degree:
            degreeCount[i] = ((endRangeCount[i]-startRangeCount[i])+1)
    return min(degreeCount.values())

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr_count = int(input().strip())

    arr = []

    for _ in range(arr_count):
        arr_item = int(input().strip())
        arr.append(arr_item)

    result = degreeOfArray(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
