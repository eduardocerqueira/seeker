#date: 2021-12-28T17:15:12Z
#url: https://api.github.com/gists/c1a8727c8e419db4d2f4dcac4c4caa97
#owner: https://api.github.com/users/tseng1026

class Solution:
    # time - O(n^2) / space - O(n)
    def numTrees(self, n: int) -> int:
        result = [1, 1]
        for total_nodes in range(2, n + 1):
            temp_result = 0
            for left_nodes in range(total_nodes):
                right_nodes = total_nodes - left_nodes - 1
                temp_result += result[left_nodes] * result[right_nodes]
            result.append(temp_result)
        return result[n]