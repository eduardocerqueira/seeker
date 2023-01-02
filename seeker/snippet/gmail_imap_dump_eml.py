#date: 2023-01-02T16:40:55Z
#url: https://api.github.com/gists/94596191338f330092e5557c9f753f7f
#owner: https://api.github.com/users/n2x4

#!/usr/bin/env python
#
# Very simple Python script to dump all emails in an IMAP folder to files.  
# This code is released into the public domain.
#
# RKI Nov 2013
#
import sys
import imaplib
import getpass

IMAP_SERVER = 'imap.gmail.com'
EMAIL_ACCOUNT = "notatallawhistleblowerIswear@gmail.com"
EMAIL_FOLDER = "**********"
OUTPUT_DIRECTORY = 'C:/src/tmp'

PASSWORD = "**********"


def process_mailbox(M):
    """
    Dump all emails in the folder to files in output directory.
    """

    rv, data = M.search(None, "ALL")
    if rv != 'OK':
        print "No messages found!"
        return

    for num in data[0].split():
        rv, data = M.fetch(num, '(RFC822)')
        if rv != 'OK':
            print "ERROR getting message", num
            return
        print "Writing message ", num
        f = open('%s/%s.eml' %(OUTPUT_DIRECTORY, num), 'wb')
        f.write(data[0][1])
        f.close()

def main():
    M = imaplib.IMAP4_SSL(IMAP_SERVER)
    M.login(EMAIL_ACCOUNT, PASSWORD)
    rv, data = M.select(EMAIL_FOLDER)
    if rv == 'OK':
        print "Processing mailbox: ", EMAIL_FOLDER
        process_mailbox(M)
        M.close()
    else:
        print "ERROR: Unable to open mailbox ", rv
    M.logout()

if __name__ == "__main__":
    main()
_main__":
    main()
