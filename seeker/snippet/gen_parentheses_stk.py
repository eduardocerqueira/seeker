#date: 2023-05-26T16:51:24Z
#url: https://api.github.com/gists/ed6ba91351dd50a9f82d616f55156d4e
#owner: https://api.github.com/users/Nasfame

from typing import List

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        # Initialize the stack with the starting values
        stack = [("(", 1, 0)]
        result = []

        # Perform iterative backtracking until the stack is empty
        while stack:
            s, left, right = stack.pop()

            # If the length of the string is equal to 2*n, add it to the result
            if len(s) == 2 * n:
                result.append(s)
            else:
                # Add the opening parenthesis if there are remaining left parentheses
                if left < n:
                    stack.append((s + "(", left + 1, right))

                # Add the closing parenthesis if there are more left parentheses than right parentheses
                if right < left:
                    stack.append((s + ")", left, right + 1))

        return result
