#date: 2021-12-31T16:40:40Z
#url: https://api.github.com/gists/378c9dc113be75a40d0b25a38c932178
#owner: https://api.github.com/users/sohel473

class Solution:
	# @param A : list of integers
	# @param B : integer
	# @return an integer
	def diffPossible(self, A, B):
        def binary_search(left, right, target):
            while left <= right:
                mid = (left + right) // 2
                if A[mid] == target:
                    return True
                elif A[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return False


        for i in range(len(A)-1):
            target = A[i] + B
            found = binary_search(i+1, len(A)-1, target)
            if found:
                return 1
        return 0