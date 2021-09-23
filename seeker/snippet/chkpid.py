#date: 2021-09-23T17:13:00Z
#url: https://api.github.com/gists/eedc05317d0b8406c2722b0e19a83384
#owner: https://api.github.com/users/wannadrunk

def chkpid(pid):
	pid = "".join(pid.split())
	thai_num = "๐๑๒๓๔๕๖๗๘๙"
	if len(pid) != 13:
		return False
	if not(pid.isdigit()):
		return False
	if pid[0] in thai_num:
		return False

	# initial variable
	index = 0
	sum = 0
	while index < 12:
		sum += int(pid[index]) * (13 - index)
		index += 1

	d13 = sum % 11
	if d13 > 1:
		d13 = 11 - d13
	else:
		d13 = 0

	if d13 == int(pid[12]):
		return True
	else:
		return False