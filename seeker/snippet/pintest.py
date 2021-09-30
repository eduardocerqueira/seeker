#date: 2021-09-30T17:00:51Z
#url: https://api.github.com/gists/f1e16e6b41fd93ca3e4fc5cad4a41dff
#owner: https://api.github.com/users/foone

import struct,sys,os
name=sys.argv[1]
class AllBitsTester(object):
	def __init__(self):
		self.allbits =0 
	def test(self, x):
		self.allbits|=x
	def completed(self):
		return self.allbits == 1023
	def message(self):
		return 'all bits active'
	def debug(self):
		return ''
class BitCorrelations(object):
	def __init__(self, left, right):
		self.leftMask=(1<<left)
		self.rightMask=(1<<right)
		self.bits=(left,right)
		self.cases=set()
	def test(self, x):
		k=(x&self.leftMask,x&self.rightMask)
		self.cases.add(k)
	def completed(self):
		return len(self.cases) == 4
	def message(self):
		return 'variations on bits {} and {}'.format(*self.bits)
	def debug(self):
		return repr(self.cases)


testers=[AllBitsTester()]
for a in range(10):
	for b in range(a):
		testers.append(BitCorrelations(a,b))
samples=0
with open(name,'rb') as f:
	f.seek(0,os.SEEK_END)
	length=f.tell()/5*4
	f.seek(0,os.SEEK_SET)
	while True:
		data=f.read(5)
		if not data:
			break
		b1, b2, b3, b4, b5 = tuple([ord(x) for x in data])
		o1 = (b1 << 2) + (b2 >> 6)
		o2 = ((b2 % 64) << 4) + (b3 >> 4)
		o3 = ((b3 % 16) << 6) + (b4 >> 2)
		o4 = ((b4 % 4) << 8) + b5
		newtesters=[]
		for tester in testers:
			for tbyte in (o1,o2,o3,o4):
				tester.test(tbyte)
			if not tester.completed():
				newtesters.append(tester)
			else:
				print 'Found {}'.format(tester.message())
		samples+=1
		testers=newtesters
		if not testers:
			print 'All testers finished after {} samples'.format(samples*4)
			break
		else:
			if (samples % 1000000) == 0:
				print 'Still waiting on testers. Position is {} of {} ({:0.1f}%)'.format(samples,length,samples*100.0/length)
				for tester in testers:
					print '* ',tester.message()
					dstr=tester.debug()
					if dstr:
						print '   ',dstr

