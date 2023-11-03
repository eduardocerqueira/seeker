#date: 2023-11-03T17:06:14Z
#url: https://api.github.com/gists/ec11f7704b8e5c1a92938a1d91286ff9
#owner: https://api.github.com/users/rajrao

import timeit
from pprint import pprint

setup = '''
from string import Template
s = "the quick brown fox JUMPED OVER THE"
t = "LAZY DOG'S BACK 1234567890"
'''
iter = 1
baseline = timeit.timeit("f'{s} {t}'", setup, number=iter)
print("%.10f" % baseline)

time = timeit.timeit("s + '  ' + t", setup, number=iter)
print("%.10f factor: %.2f" % (time, time/baseline))

time = timeit.timeit("' '.join((s, t))", setup, number=iter)
print("%.10f factor: %.2f" % (time, time/baseline))

time = timeit.timeit("'%s %s' % (s, t)", setup, number=iter)
print("%.10f factor: %.2f" % (time, time/baseline))

time = timeit.timeit("'{} {}'.format(s, t)", setup, number=iter)
print("%.10f factor: %.2f" % (time, time/baseline))

time = timeit.timeit("Template('$s $t').substitute(s=s, t=t)", setup, number=iter)
print("%.10f factor: %.2f" % (time, time/baseline))
