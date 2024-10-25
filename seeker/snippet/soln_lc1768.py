#date: 2024-10-25T16:53:51Z
#url: https://api.github.com/gists/791547ee832241a01c0bf92725f73c39
#owner: https://api.github.com/users/hongtaoh


class Solution:
  """
  This is okay but unnecessarily complicated. 
  if a = [1, 2, 3], then a[4:] won't cause any errors. This will help me get a better solution.
  """
    def mergeAlternately(self, word1: str, word2: str) -> str:
        # len1 < len2
        len1, len2 = (len(word1), len(word2)) if len(word1) < len(word2) else (len(word2), len(word1))
        output = []
        for i in range(len1):
            output.append(word1[i])
            output.append(word2[i])
        if len1 == len(word1):
            output.append(word2[len1:])
        elif len1 != len2:
            output.append(word1[len1:])
        return "".join(output)