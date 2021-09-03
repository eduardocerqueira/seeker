#date: 2021-09-03T17:13:37Z
#url: https://api.github.com/gists/f82708142ebb84d0e3b6b76456fe6047
#owner: https://api.github.com/users/Maine558

def find_short(s):
    l = 1000000000000
    s = s.split()
    for i in range(len(s)):
        print(len(s[i]))
        q = len(s[i])
        print(q)
        if l > q:
            l = q
    print(l)
    return l # l: shortest word length

find_short("bitcoin take over the world maybe who knows perhaps")