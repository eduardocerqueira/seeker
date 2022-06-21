#date: 2022-06-21T17:26:10Z
#url: https://api.github.com/gists/038429f94dfaa3323276a298671fb58e
#owner: https://api.github.com/users/p1aton

import smtplib
from email.mime.text import MIMEText

smtp_ssl_host = 'smtp.gmail.com'  # smtp.mail.yahoo.com
smtp_ssl_port = 465
username = 'USERNAME or EMAIL ADDRESS'
password = 'PASSWORD'
sender = 'ME@EXAMPLE.COM'
targets = ['HE@EXAMPLE.COM', 'SHE@EXAMPLE.COM']

msg = MIMEText('Hi, how are you today?')
msg['Subject'] = 'Hello'
msg['From'] = sender
msg['To'] = ', '.join(targets)

server = smtplib.SMTP_SSL(smtp_ssl_host, smtp_ssl_port)
server.login(username, password)
server.sendmail(sender, targets, msg.as_string())
server.quit()
