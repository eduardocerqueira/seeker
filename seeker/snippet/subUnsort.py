#date: 2021-11-05T16:56:37Z
#url: https://api.github.com/gists/b2baa80f75323be5867539bae8486748
#owner: https://api.github.com/users/sohel473

class Solution:
	# @param A : list of integers
	# @return a list of integers
	def subUnsort(self, A):
        B = sorted(A)

        i, j = 0, len(A)-1
        start, end = 0, 0

        while i < len(A):
            if A[i] != B[i]:
                start = i
                break
            i += 1
        if A[start] == B[end]:
            return [-1]
        while j > 0:
            if A[j] != B[j]:
                end = j
                break
            j -= 1
        return [start, end]