#date: 2023-01-27T16:47:36Z
#url: https://api.github.com/gists/b2a50f74d2deebfdcbeef5856905dfdc
#owner: https://api.github.com/users/Sidheshwarj2001

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def sendMail():
    fromEmail = "______@gmail.com"
    EmailPawword = "evogucrtwpjcknco"
    sendToEmail = "Sidjawale007@gmail.com"

    # making instance of multipart

    msg = MIMEMultipart()

    msg["From"] = fromEmail
    msg["To"] = sendToEmail

    msg["Subject"] = " This is mail with attach file"

    body = " Body of the mail send by python script"

    # open the file which you  want to send
    filename = "file name with extension"
    attachment = open("first.py",'rb')

    msg.attach(MIMEText(body, "plain"))

    # instance of MINEBase and named as p
    p = MIMEBase('application', 'octet-stream')

    p.set_payload(attachment.read())  # file attach

    encoders.encode_base64(p)

    p.add_header('current-Disponsition', 'attachment; filename = %s' % filename)

    msg.attach(p)

    server = smtplib.SMTP("smtp.gmail.com", 587)

    server.starttls()

    server.login(fromEmail,EmailPawword)

    text = msg.as_string()

    server.sendmail(fromEmail,sendToEmail,text)

    server.quit()

    print("mail successfully send")


def main():
    sendMail()

if __name__ == "__main__":
    main()