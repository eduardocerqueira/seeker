#date: 2023-04-17T16:58:57Z
#url: https://api.github.com/gists/0c935dc5ecf77990395e70a40c7ead10
#owner: https://api.github.com/users/saulfm08

#!/usr/bin/env python3

SMTPserver = 'email-smtp.us-east-1.amazonaws.com'
sender =     'the-email-sender-here@domain.example.com'
destination = ['one-email-here@domain.example.com', 'other-email-here@domain.example.com', 'another-email-here@domain.example.com']

USERNAME = "your smtp user here"
PASSWORD = "**********"

# typical values for text_subtype are plain, html, xml
text_subtype = 'plain'


content="""\
Test message here
"""

subject="I am Testing SES - Sent from Python"

from smtplib import SMTP_SSL as SMTP       # this invokes the secure SMTP protocol (port 465, uses SSL)
from email.mime.text import MIMEText

try:
    msg = MIMEText(content, text_subtype)
    msg['Subject']= subject
    msg['From']   = sender # some SMTP servers will do this automatically, not all
    msg['To'] = ", ".join(destination)

    print(str(msg))

    conn = SMTP(SMTPserver)
    conn.set_debuglevel(False)
    conn.login(USERNAME, PASSWORD)
    try:
        conn.sendmail(sender, destination, msg.as_string())
    finally:
        conn.quit()
    print("Success!")

except Exception as ex:
    print(ex)
   print(ex)
