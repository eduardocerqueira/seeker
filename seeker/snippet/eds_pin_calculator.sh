#date: 2023-02-16T17:10:24Z
#url: https://api.github.com/gists/8f371cda74b7d4d79a7ea376935edd5f
#owner: https://api.github.com/users/besmirzanaj

#!/bin/bash

# EDS Certificate pin calculator
# 2023, Besmir Zanaj

# Usage:
# /ets_pin_calculator.sh <CERT_FILE>

# This script will calulate the EDS cert pin for a certificate file.
# More info in RFC-7469 - https://www.rfc-editor.org/rfc/rfc7469#section-2.1.1

# It takes only one argument, the CERT File in PEM (x509) format.
# Make sure openssl binary is already installed or available in $PATH

openssl x509 -in $1 -noout -pubkey | openssl pkey -inform PEM -pubin -outform der | openssl sha1 -sha256 -binary | openssl enc -base64