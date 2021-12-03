#date: 2021-12-03T17:11:08Z
#url: https://api.github.com/gists/3a88bec217c9016564f12d7b11245a83
#owner: https://api.github.com/users/Mamatha-Aluvala

import heapq

def maxKProfit(n, b, arr):
    arr = [-1 * i for i in arr]
    heapq.heapify(arr)
    profit = 0
    while b:
        soldat = -1 * heapq.heappop(arr)
        profit += soldat
        heapq.heappush(arr, 1 - soldat)
        b -= 1
    return profit

def main():
    n, b = list(map(int, input().strip().split(' ')))
    arr = list(map(int, input().strip().split(' ')))

    maxProfit = maxKProfit(n, b, arr)
    print(maxProfit)

if __name__=="__main__":
    main()


# Crio Methodology
  
# Milestone 1: Understand the problem clearly
# 1. Ask questions & clarify the problem statement clearly.
# 2. Take an example or two to confirm your understanding of the input/output & extend it to test cases
# Milestone 2: Finalize approach & execution plan
# 1. Understand what type of problem you are solving.
#      a. Obvious logic, tests ability to convert logic to code
#      b. Figuring out logic
#      c. Knowledge of specific domain or concepts
#      d. Knowledge of specific algorithm
#      e. Mapping real world into abstract concepts/data structures
# 2. Brainstorm multiple ways to solve the problem and pick one
# 3. Get to a point where you can explain your approach to a 10 year old
# 4. Take a stab at the high-level logic & write it down.
# 5. Try to offload processing to functions & keeping your main code small.
# Milestone 3: Code by expanding your pseudocode
# 1. Make sure you name the variables, functions clearly.
# 2. Avoid constants in your code unless necessary; go for generic functions, you can use examples for your thinking though.
# 3. Use libraries as much as possible
# Milestone 4: Prove to the interviewer that your code works with unit tests
# 1. Make sure you check boundary conditions
# 2. Time & storage complexity
# 3. Suggest optimizations if applicable