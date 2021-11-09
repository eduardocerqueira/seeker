#date: 2021-11-09T17:08:11Z
#url: https://api.github.com/gists/a8cee3b7185139c30c5fe8271e5a2607
#owner: https://api.github.com/users/matt4791

#!/home/marcload/venv/bin/python

import paramiko
import smtplib
import os.path
import glob
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders


def create_sftp_client(host, port, username, password):
	ssh = None
	sftp = None

	try:

		ssh = paramiko.SSHClient()
		ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		ssh.banner_timeout = 200
		ssh.connect(host, port, username, password)

		sftp = ssh.open_sftp()
		sftp.sshclient = ssh

		return sftp
	except Exception as e:

		print('An error occurred creating SFTP client: %s: %s' % (e.__class__, e))
		if sftp is not None:
			sftp.close()
		if ssh is not None:
			ssh.close()
		pass


def send(email_sender, email_from, email_to, email_reply_to, email_subject, email_body, attachment):
	message = MIMEMultipart()
	message['From'] = email_from
	message['To'] = email_to
	message['Bcc'] = 'support@ohionet.org'
	message['Subject'] = email_subject
	message['Date'] = formatdate(localtime=True)
	message['Reply-To'] = email_reply_to
	message.attach(MIMEText(email_body))

	spreadsheet = MIMEBase('application', 'octet-stream')
	spreadsheet.set_payload(open(attachment, 'rb').read())
	encoders.encode_base64(spreadsheet)
	spreadsheet.add_header('Content-Disposition', 'attachment; filename="' + os.path.basename(attachment) + '"')

	message.attach(spreadsheet)

	email = smtplib.SMTP('localhost')
	email.sendmail(email_sender, email_to, message.as_string())


def find_newest_file(path):
	# print('Path: ' + path)
	list_of_files = glob.glob(str(path + '/*'))
	# print('Files: ' + str(list_of_files))
	# only keep files, not directories
	list_of_files = [x for x in list_of_files if os.path.isfile(x)]
	# print('Files: ' + str(list_of_files))
	if not list_of_files:
		return None
	newest_file = max(list_of_files, key=os.path.getmtime)
	# print('Newest file: ' + newest_file)
	return newest_file


def get_ymd_date_from_timestamp(file_name):
	if os.path.exists(file_name):
		m_timestamp = os.path.getmtime(file_name)
		last_mod = datetime.fromtimestamp(m_timestamp)
		stamp = last_mod.strftime('%Y%m%d')
	else:
		stamp = 'XXXXXXXX'

	return stamp


def get_timestamp(file_name):
	if os.path.exists(file_name):
		m_timestamp = os.path.getmtime(file_name)
		last_mod = datetime.fromtimestamp(m_timestamp)
		stamp = last_mod.strftime('%Y%m%d%H%M%S')
	else:
		stamp = 'XXXXXXXXxxxxxx'

	return stamp