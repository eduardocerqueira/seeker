#date: 2022-09-08T17:21:16Z
#url: https://api.github.com/gists/5a68fe2f7dd412b06fa73fa15fe7d17c
#owner: https://api.github.com/users/mateidanut

class Solution:
    def thousandSeparator(self, n: int) -> str:
        inverted_s = str(n)[::-1]
        lst = []
        for i, c in enumerate(inverted_s):
            if i % 3 == 0:
                lst.append('.')
            lst.append(c)
            
        normal_s = ''.join(lst[1:])[::-1]
        return normal_s