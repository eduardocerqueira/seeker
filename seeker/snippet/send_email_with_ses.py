#date: 2023-08-10T17:03:15Z
#url: https://api.github.com/gists/1e0058e15c8768082c7dca34a5597dfc
#owner: https://api.github.com/users/vg-vaibhav

import os
import ssl
from smtplib import SMTP
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText


def send_email_with_attachement_with_ses(sender, recievers, subject, message, filename, attachment_name):
    # getting the credentials fron evironemnt
    host = os.environ.get("SES_HOST_ADDRESS")
    port = os.environ.get("SES_PORT")
    user = os.environ.get("SES_USER_ID")
    password = "**********"

    # setting up ssl context
    context = ssl.create_default_context()
    
    # Create a multipart message
    msg = MIMEMultipart()
    body_part = MIMEText(message, "plain")
    msg["Subject"] = subject
    msg["From"] = sender
    # msg["To"] = receivers
    
    msg.attach(body_part)
    # open and read the CSV file in binary
    with open(filename, "rb") as file:
        # Attach the file with filename to the email
        msg.attach(MIMEApplication(file.read(), Name=attachment_name))
    
    # creating an unsecure smtp connection
    with SMTP(host, port) as server:
        # securing using tls
        server.starttls(context=context)
        # authenticating with the server to prove our identity
        server.login(
            user=user,
            password= "**********"
        )
        # sending a plain text email
        server.sendmail(sender, recievers, msg.as_string())
        server.quit()
server.quit()
