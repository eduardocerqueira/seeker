#date: 2022-01-03T17:13:51Z
#url: https://api.github.com/gists/6315ace6fb3a11565b77f52e3090fcf6
#owner: https://api.github.com/users/byzantic


def ana_key(word):
    return ''.join(sorted(word))

def addAna(anas, word):
        key = ana_key(word)
        ws  = anas.get(key,set())
        ws.add(word)
        anas[key] = ws

def fromList(wlist):
    anas = {}
    for w in wlist :
        addAna(anas,w)
    return anas

def get_anagrams(anas, word):
    return anas.get(ana_key(word), None)

# example
anas = fromList(['step','pets','pest','swallow','wallows','rebut','rebuke','owl','low'])

# should return {'swallow','wallows'}
get_anagrams(anas,'swallow')