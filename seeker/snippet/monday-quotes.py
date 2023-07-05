#date: 2023-07-05T16:44:37Z
#url: https://api.github.com/gists/9d214982de77b616b995aab055ad2036
#owner: https://api.github.com/users/Phil-Miles

import smtplib
import datetime as dt
from random import choice

server = 'smtp-mail.outlook.com'
port = 587

sender_email = 'glorobottest@hotmail.com'
password = "**********"
recipient = 'recipient_email@gmail.com'
quotes_list = ['quote1',
               'quote2',
               'quote3']

now = dt.datetime.now()

if now.weekday() == 0:
    subject = "Happy Monday!"
    message = choice(quotes_list)
    email = f'Subject: {subject}\n\n{message}'
    with smtplib.SMTP(server, port) as connection:
        connection.starttls()
        connection.login(user= "**********"=password)
        connection.sendmail(sender_email, recipient, email)mail(sender_email, recipient, email)