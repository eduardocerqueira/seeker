#date: 2023-07-19T16:58:14Z
#url: https://api.github.com/gists/616d723dbc88cb466470b3f3cf79dca0
#owner: https://api.github.com/users/KevinHonka

#!/usr/bin/env python
#
# Adjusted the python script from https://gist.github.com/robulouski/7442321
# This is now better suited for python3 environments.
#

import argparse
import imaplib
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")
log = logging.getLogger(__name__)


def process_mailbox(mail_connection, output_dir):
    """
    Dump all emails in the folder to files in output directory.
    """

    rv, data = mail_connection.search(None, "ALL")
    if rv != "OK":
        log.info("No messages found")
        sys.exit(0)

    for num in data[0].split():
        rv, data = mail_connection.fetch(num, "(RFC822)")
        if rv != "OK":
            log.error(f"Failure getting message { num.decode('ascii') }")
            return
        log.debug(f"Writing message { num.decode('ascii') }")
        f = open("%s/%s.eml" % (output_dir, num.decode("ascii")), "wb")
        f.write(data[0][1])
        f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("MAIL_HOST", "localhost"))
    parser.add_argument(
        "--ssl", default=os.getenv("MAIL_SSLMODE", False), action="store_true"
    )
    parser.add_argument(
        "--debug", default=os.getenv("MAIL_DEBUG", False), action="store_true"
    )
    parser.add_argument("--account", default=os.getenv("MAIL_ACCOUNT"))
    parser.add_argument("--maildir", default=os.getenv("MAIL_DIR", "INBOX"))
    parser.add_argument("--output", default=os.getenv("MAIL_OUTPUT_DIR", "/tmp/"))
    parser.add_argument("--password", default= "**********"

    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    if args.ssl:
        mail_connection = imaplib.IMAP4_SSL(args.host)
    else:
        mail_connection = imaplib.IMAP4(args.host)

    mail_connection.login(args.account, args.password)
    rv, data = mail_connection.select(args.maildir)
    if rv == "OK":
        log.info(f"Processing mailbox: { args.maildir }")
        process_mailbox(mail_connection, args.output)
        mail_connection.close()
    else:
        log.error(f"Unable to open mailbox { rv }")
    mail_connection.logout()


if __name__ == "__main__":
    main()
":
    main()
