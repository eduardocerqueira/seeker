#date: 2023-07-28T16:41:44Z
#url: https://api.github.com/gists/e52f8a95eb2ca21d7117ef94cb8e8f60
#owner: https://api.github.com/users/abhiphile

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        dic=dict()
        res=[]
        lis=[]
        for i in nums:
            if i in dic:
                dic[i]+=1
            else:
                dic[i]=1
        for i in dic.values():
            lis.append(i)
        lis.sort()
        lis=lis[-k:]
        for i in lis[:k]:
            for j in dic:
                if dic[j]==i and j not in res:
                    res.append(j)
        return res
        


        