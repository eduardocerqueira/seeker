#date: 2023-05-26T16:51:24Z
#url: https://api.github.com/gists/ed6ba91351dd50a9f82d616f55156d4e
#owner: https://api.github.com/users/Nasfame

class Solution:

    n: int

    def backtrack(self, s:str="", left:int=0, right:int=0,combs:List[str]=None) :
        if combs is None:
            combs = []

        n = self.n

        if len(s)==2*n:
            combs.append(s)
            return

        if left<n:
            self.backtrack(s+"(",left+1,right,combs)
        
        if right<left:
            self.backtrack(s+")",left,right+1,combs)

        return combs


    def generateParenthesis(self, n: int) -> List[str]:
        # O(10^n)
        self.n = n

        res = self.backtrack()

        return res