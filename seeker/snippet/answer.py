#date: 2023-09-11T17:06:59Z
#url: https://api.github.com/gists/127816860b1de2fe02ee49d877d78180
#owner: https://api.github.com/users/SahilPachlore

import sys

def calculate_sum(num_test_cases, current_test_case):
    if current_test_case > num_test_cases:
        return []

    num_integers = int(input())
    integers = list(map(int, input().split()))

    test_case_sum = sum(x ** 2 for x in integers if x >= 0)
    return [test_case_sum] + calculate_sum(num_test_cases, current_test_case + 1)

def main():
    num_test_cases = int(input())
    sys.setrecursionlimit(num_test_cases + 10)  # Set recursion limit

    results = calculate_sum(num_test_cases, 1)

    for result in results:
        print(result)

if __name__ == "__main__":
    main()