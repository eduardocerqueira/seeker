#date: 2023-06-26T16:58:17Z
#url: https://api.github.com/gists/706375d0b18e63ffa906acd8c13fc4b0
#owner: https://api.github.com/users/markbrutx

def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

    res = []
    def backtrack(remain, cur, start):

        if remain == 0:
            res.append(list(cur))
            return
        elif remain < 0:
            return
        else:
            for i in range(start, len(candidates)):
                cur.append(candidates[i])
                backtrack(remain - candidates[i], cur, i)

                cur.pop()
    backtrack(target, [], 0)
    return res