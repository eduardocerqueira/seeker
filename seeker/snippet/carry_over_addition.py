#date: 2021-09-07T16:52:32Z
#url: https://api.github.com/gists/910330e1bc10c14a10bce2203feaf3ef
#owner: https://api.github.com/users/codecakes


class Solution:
    def addStrings(self, s1, s2):
        '''
        :type s1: str
        :type s2: str
        :rtype: str
        '''
        i = s1_len = len(s1) - 1
        j = s2_len = len(s2) - 1
        digit = 1
        ans = 0
        while i >= 0 or j >= 0:
            res = 0
            if i >= 0:
                res = ord(s1[i]) - ord("0")
                # print(f"s1={res}")
                i -= 1
            if j >= 0:
                res += ord(s2[j]) - ord("0")
                # print(f"s2={res}")
                j -= 1
            carry = res//10
            res %= 10
            ans += (carry * 10**digit) + (res * 10**(digit-1))
            # print(f"carry={carry} res={res} ans={ans}")
            digit += 1
        return str(ans)
            
