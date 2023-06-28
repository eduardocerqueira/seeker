#date: 2023-06-28T16:46:59Z
#url: https://api.github.com/gists/6f33a923369bac3ccfd12fa91a725fe5
#owner: https://api.github.com/users/elicharlese

class Solution(object):
    def decodeString(self, s):
        stack = []; curNum = 0; curString = ''
        for c in s:
            if c == '[':
                stack.append(curString)
                stack.append(curNum)
                curString = ''
                curNum = 0
            elif c == ']':
                num = stack.pop()
                prevString = stack.pop()
                curString = prevString + num*curString
            elif c.isdigit():
                curNum = curNum*10 + int(c)
            else:
                curString += c
        return curString