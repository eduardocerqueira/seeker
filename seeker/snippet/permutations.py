#date: 2023-06-27T17:00:58Z
#url: https://api.github.com/gists/580609e8cd5f10cf433fba1e8b43f9d4
#owner: https://api.github.com/users/markbrutx

def permute(self, nums: List[int]) -> List[List[int]]:
    res = []
    if len(nums) == 1:
        return [nums[:]]

    for i in range(len(nums)):
        n = nums.pop(0)
        perms = self.permute(nums)

        for perm in perms:
            perm.append(n)
        res.extend(perms)
        nums.append(n)


    return res