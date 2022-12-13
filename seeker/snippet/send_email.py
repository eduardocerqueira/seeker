#date: 2022-12-13T16:54:50Z
#url: https://api.github.com/gists/d818b58bd35ee45d68e3afb1128ca391
#owner: https://api.github.com/users/swapnanilsharma

def sendEmail(receiver, cc, subject, message, filename,
              bcc, sender="test@gmail.com"):
    data = MIMEMultipart()
    data['From'] = sender
    data['To'] = receiver
    data['Cc'] = cc
    data['Bcc'] = bcc
    data['Subject'] = subject
    data.attach(MIMEText(message, 'html'))
    try:
        attachment = open("{}".format(filename), "rb")
        p = MIMEBase('application', 'octet-stream')
        p.set_payload((attachment).read())
        encoders.encode_base64(p)
        p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        data.attach(p)
    except FileNotFoundError:
        pass
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(sender, "password")
    text = data.as_string()
    s.sendmail(receiver, sender, text)
    s.quit()