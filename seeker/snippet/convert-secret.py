#date: 2023-04-17T17:05:57Z
#url: https://api.github.com/gists/112ce89c22385211d5620f9deffe64f1
#owner: https://api.github.com/users/saulfm08

#!/usr/bin/env python3

import hmac
import hashlib
import base64
import argparse

SMTP_REGIONS = [
    'us-east-2',       # US East (Ohio)
    'us-east-1',       # US East (N. Virginia)
    'us-west-2',       # US West (Oregon)
    'ap-south-1',      # Asia Pacific (Mumbai)
    'ap-northeast-2',  # Asia Pacific (Seoul)
    'ap-southeast-1',  # Asia Pacific (Singapore)
    'ap-southeast-2',  # Asia Pacific (Sydney)
    'ap-northeast-1',  # Asia Pacific (Tokyo)
    'ca-central-1',    # Canada (Central)
    'eu-central-1',    # Europe (Frankfurt)
    'eu-west-1',       # Europe (Ireland)
    'eu-west-2',       # Europe (London)
    'sa-east-1',       # South America (Sao Paulo)
    'us-gov-west-1',   # AWS GovCloud (US)
]

# These values are required to calculate the signature. Do not change them.
DATE = "11111111"
SERVICE = "ses"
MESSAGE = "SendRawEmail"
TERMINAL = "aws4_request"
VERSION = 0x04


def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


 "**********"d "**********"e "**********"f "**********"  "**********"c "**********"a "**********"l "**********"c "**********"u "**********"l "**********"a "**********"t "**********"e "**********"_ "**********"k "**********"e "**********"y "**********"( "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"k "**********"e "**********"y "**********", "**********"  "**********"r "**********"e "**********"g "**********"i "**********"o "**********"n "**********") "**********": "**********"
    if region not in SMTP_REGIONS:
        raise ValueError(f"The {region} Region doesn't have an SMTP endpoint.")

    signature = "**********"
    signature = sign(signature, region)
    signature = sign(signature, SERVICE)
    signature = sign(signature, TERMINAL)
    signature = sign(signature, MESSAGE)
    signature_and_version = bytes([VERSION]) + signature
    smtp_password = "**********"
    return smtp_password.decode('utf-8')


def main():
    parser = argparse.ArgumentParser(
        description= "**********"
    parser.add_argument(
        'secret', help= "**********"
    parser.add_argument(
        'region',
        help= "**********"
        choices=SMTP_REGIONS)
    args = parser.parse_args()
    print(calculate_key(args.secret, args.region))


if __name__ == '__main__':
    main()
