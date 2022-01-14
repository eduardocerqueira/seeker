#date: 2022-01-14T16:59:46Z
#url: https://api.github.com/gists/eee8e3281fafd0cb2401e7764673c266
#owner: https://api.github.com/users/sahasourav17

from collections import Counter

def anagrams(s1,s2):
    """
    As seen in method 2, we first check if both strings have the 
    same length because if they're not it's impossible for them
    to be anagrams.
    """
    if len(s1) != len(s2):
        return False

    #comparing dictionaries with equality operator
    return Counter(s1) == Counter(s2)