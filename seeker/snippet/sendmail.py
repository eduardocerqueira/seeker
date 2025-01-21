#date: 2025-01-21T17:01:29Z
#url: https://api.github.com/gists/84dffd3357f9f9a17ff865d4cc804a40
#owner: https://api.github.com/users/NiceRath

#!/usr/bin/env python3

import smtplib
from email.message import EmailMessage
from pathlib import Path
from os.path import basename
from argparse import ArgumentParser

REQUIRE_FROM_DOMAIN = '@test.com'


def main():
    msg = EmailMessage()
    msg['Subject'] = args.subject
    msg['From'] = getattr(args, 'from')
    msg['To'] = args.to
    msg.set_content(args.body, subtype='html')

    with open(args.attachment, 'rb') as f:
        file_data = f.read()

    msg.add_attachment(
        file_data, maintype='text', subtype='plain',
        filename=args.attachment_name if args.attachment_name is not None else basename(args.attachment),
    )

    with smtplib.SMTP('127.0.0.1', 25) as server:
        server.send_message(msg)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-f', '--from', required=False,
        default='test@test.com', help='From E-Mail Address',
    )
    parser.add_argument('-t', '--to', required=True, help='To E-Mail Address')
    parser.add_argument('-a', '--attachment', required=True, help='Path to attachment')
    parser.add_argument('-an', '--attachment-name', required=False, help='Name of attachment')
    parser.add_argument('-b', '--body', required=True, help='E-Mail Body')
    parser.add_argument('-s', '--subject', required=True, help='E-Mail Subject')
    args = parser.parse_args()

    if not Path(args.attachment).is_file():
        raise FileNotFoundError(f'No such file: {args.attachment}')

    if getattr(args, 'from').find(REQUIRE_FROM_DOMAIN) == -1:
        raise FileNotFoundError(f'From domain needs to be: {REQUIRE_FROM_DOMAIN}')

    main()
