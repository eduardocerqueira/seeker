#date: 2023-03-13T16:51:39Z
#url: https://api.github.com/gists/67b1a4defacaacb12a4197cbd3d91c8b
#owner: https://api.github.com/users/FilipRazek

def build_message(sender, recipient, subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    return msg