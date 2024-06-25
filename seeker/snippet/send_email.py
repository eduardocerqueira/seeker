#date: 2024-06-25T16:34:28Z
#url: https://api.github.com/gists/890c6124dbc174fc461525603d7cc87b
#owner: https://api.github.com/users/isaachilly

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import argparse

def print_ascii_email(from_email, to_email, subject, body, smtp_server, smtp_port, force_tls):
    ascii_art = f"""
+----------------------------------------------------------+
|                       EMAIL TERMINAL                     |
+----------------------------------------------------------+
| From: {from_email:<51}|
| To: {to_email:<53}|
| Subject: {subject:<48}|
| Body: {body:<51}|
|                                                          |
| SMTP Server: {smtp_server:<44}|
| SMTP Port: {smtp_port:<46}|
| Force TLS: {str(force_tls):<46}|
+----------------------------------------------------------+
|                 Sending email... Please wait.            |
+----------------------------------------------------------+
"""
    print(ascii_art)

 "**********"d "**********"e "**********"f "**********"  "**********"s "**********"e "**********"n "**********"d "**********"_ "**********"e "**********"m "**********"a "**********"i "**********"l "**********"( "**********"s "**********"m "**********"t "**********"p "**********"_ "**********"s "**********"e "**********"r "**********"v "**********"e "**********"r "**********", "**********"  "**********"s "**********"m "**********"t "**********"p "**********"_ "**********"p "**********"o "**********"r "**********"t "**********", "**********"  "**********"f "**********"o "**********"r "**********"c "**********"e "**********"_ "**********"t "**********"l "**********"s "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********", "**********"  "**********"f "**********"r "**********"o "**********"m "**********"_ "**********"e "**********"m "**********"a "**********"i "**********"l "**********", "**********"  "**********"t "**********"o "**********"_ "**********"e "**********"m "**********"a "**********"i "**********"l "**********", "**********"  "**********"s "**********"u "**********"b "**********"j "**********"e "**********"c "**********"t "**********", "**********"  "**********"b "**********"o "**********"d "**********"y "**********") "**********": "**********"

    print_ascii_email(from_email, to_email, subject, body, smtp_server, smtp_port, force_tls)


    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Set up the server
        server = smtplib.SMTP(smtp_server, smtp_port)
        
        # Force TLS
        if smtp_port == 587 or smtp_port == 465 or force_tls:
            server.starttls()
        else:
            try:
                server.starttls()
            except smtplib.SMTPException:
                print("INFO: TLS is not supported on this server. Continuing without TLS encryption.")
                
        

        # Login to the server
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"  "**********"a "**********"n "**********"d "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********": "**********"
            server.login(username, password)

        # Send the email
        server.sendmail(from_email, to_email, msg.as_string())

        # Close the server connection
        server.quit()

        print("INFO: Email sent successfully!")

    except Exception as e:
        print(f"ERROR: Failed to send email. Error: {e}")

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Send an email.')
    parser.add_argument('--smtp_server', type=str, help='The SMTP server address.', required=True)
    parser.add_argument('--smtp_port', type=int, help='The SMTP server port.', required=True)
    parser.add_argument('--force_tls', action='store_const', const=1, default=0, help='Set this flag to force TLS. When set, its value will be 1.')
    parser.add_argument('--username', type=str, help='Your email username.', required=False)
    parser.add_argument('--password', type= "**********"='Your email password.', required=False)
    parser.add_argument('--from_email', type=str, help='The email address you are sending the email from.', required=True)
    parser.add_argument('--to_email', type=str, help='The email address you are sending the email to.', required=True)
    parser.add_argument('--subject', type=str, help='The subject of the email.', required=True)
    parser.add_argument('--body', type=str, help='The body of the email.', required=True)
    

    args = parser.parse_args()
    smtp_server = args.smtp_server
    smtp_port = args.smtp_port
    force_tls = args.force_tls
    username = args.username
    password = "**********"
    from_email = args.from_email
    to_email = args.to_email
    subject = args.subject
    body = args.body


    send_email(smtp_server, smtp_port, force_tls, username, password, from_email, to_email, subject, body)


# Example usage:
# python send_email.py --smtp_server smtp.example.com --smtp_port 25 --from_email xxx.yyy@stfc.ac.uk --to_email yyy.zzz@stfc.ac.uk --subject "Test Email" --body "This is a test email."