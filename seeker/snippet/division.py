#date: 2023-01-02T17:04:38Z
#url: https://api.github.com/gists/7aaa7498c011219e0ca67da06ff894b8
#owner: https://api.github.com/users/Baelfire18

def manual_division(a, b, digits=4):
    ans = str(a // b)
    a = a % b
    if a == 0:
        return ans
    ans += "."
    n = 0
    while n < digits and a > 0:
        a = a % b * 10
        n += 1
        ans += str(a // b)
    return ans


c = manual_division(20, 6)
print(c)
