#date: 2021-09-28T17:02:24Z
#url: https://api.github.com/gists/fcbe82f06608354dcb41a69b88dd50bd
#owner: https://api.github.com/users/Soulwest

import logging
import sys
import re

logging.basicConfig(level=logging.DEBUG) 
logging.info('Convert shadow to convertful config format')

print("Enter rules: ")
rule = '\n'.join(iter(input, ''))
logging.info('Rules: '+rule)
lines = rule.split("\n")

for line in lines:
	# Clean and format input
	line = re.sub(r'VM\d+:\d+ ', '', line)
	args = line.replace(', ', ',').split()

	if  not args:
		logging.warning('Empty shadow')
		sys.exit(1)
		
	for i, arg in enumerate(args):
		args[i] = arg.replace('px', '')	

	print('''[
	'dx' => {1},
	'dy' => {2},
	'blur' => {3},
	'color' => '{0}',
],'''.format(*args))