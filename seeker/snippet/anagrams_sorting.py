#date: 2022-01-14T17:20:26Z
#url: https://api.github.com/gists/a91de2043e114874e6cab6a0f772a74d
#owner: https://api.github.com/users/sahasourav17

def anagrams(s1,s2):
    if len(s1) != len(s2):
        return False
    return sorted(s1) == sorted(s2)