#date: 2022-12-16T16:55:15Z
#url: https://api.github.com/gists/6bd186429d7f5008026f600634cc2f97
#owner: https://api.github.com/users/ibrataha8

def echanger_mots(s):
  mots = s.split()
  mots[0], mots[-1], = mots[-1], mots[0]
  s2 = ' '.join(mots)
  return s2
s = input('Donner un phrase :')
s2 = echanger_mots(s)
print(s2) 