#date: 2022-09-27T17:09:10Z
#url: https://api.github.com/gists/e413f5033515f20a9edd3464338c26ff
#owner: https://api.github.com/users/Anooppandikashala

def doProgram(nums: list):
    ret = {}
    nums.sort(reverse=True)
    for n in nums:
        if n in ret.keys():
            x = ret[n]
            ret[n] = x + 1
        else:
            ret[n] = 1

    retRes = dict(sorted(ret.items(), key=lambda item: item[1]))
    # print(retRes)

    retList = [x for x in retRes.keys()]
    # print(retList)
    res = []
    for a in retList:
        n = ret[a]
        for i in range(n):
            res.append(a)
    print(res)


def main():
    nums1 = [1, 1, 2, 2, 2, 3]
    doProgram(nums1)
    nums2 = [2, 3, 1, 3, 2]
    doProgram(nums2)
    nums3 = [-1, 1, -6, 4, 5, -6, 1, 4, 1]
    doProgram(nums3)


if __name__ == '__main__':
    main()
