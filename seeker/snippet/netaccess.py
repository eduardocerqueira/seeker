#date: 2021-08-31T01:15:04Z
#url: https://api.github.com/gists/5f1c462ad858138febdd6e5ccad9c0e5
#owner: https://api.github.com/users/sanjeed5

#!/usr/bin/env python
import requests
import sys
import getpass
username = input('Enter username: ')
pa = str(getpass.getpass(prompt='Enter password: '))

with requests.session() as s:
	try:
		page = s.post('https://netaccess.iitm.ac.in/account/login' , data={ 'userLogin':username, 'userPassword':pa, 'submit':''}, verify=False)
	except Exception as err:
		print('Authentication Error: \n', err)

	if page.text.find('MAC') != -1:
		try:
			approve = s.post('https://netaccess.iitm.ac.in/account/approve', data = { 'duration':'2' , 'approveBtn':''}, verify=False)
		except Exception as err:
			print('Approve error: \n', err)

		if approve.text.find('MAC') != -1:
			print('done')
		else:
			print('error')
	else:
		print('Authentication Error')
	sys.exit()


'''
-------
 Usage
-------
NB : You have to install requests in python by using command ` sudo pip install requests ` in terminal.

1. Change your username in the variable
2. Give permissions for the python script to execute independently by typing ` chmod +x netaccess.py ` in terminal.
3. Execute typing ` netaccess.py `
4. Type your password when prompted
5. Add the script folder to $PATH variable in ~/.bashrc for executing the script everywhere irrespective of the working directory. 
   In the end you should be able to get the thing working typing ` netaccess.py ` in terminal.

'''