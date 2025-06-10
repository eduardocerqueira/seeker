#date: 2025-06-10T16:57:16Z
#url: https://api.github.com/gists/1fa46c3ffe8eca55099361e05a3fff0d
#owner: https://api.github.com/users/meerkat-monkey

#!/bin/env bash

################################################################################
# Wrap and import an Ethereum private key for AWS KMS
#
# Usage:
#   ./aws-wrap-and-import-eoa-private-key.sh <private-key-file> <aws-kms-key-id>
#
################################################################################

################################################################################
# Step 1: Validate input
################################################################################

# Check if both required parameters are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <private-key-file> <aws-kms-key-id>"
    exit 1
fi

private_key_file="$1"
aws_key_id="$2"

# Check if the private key file exists
if [ ! -f "$private_key_file" ]; then
    echo "Error: Private key file not found: $private_key_file"
    exit 1
fi

# Read Ethereum private key from file
eth_key=$(cat "$private_key_file" | tr -d '\n\r ')

# Validate eth_key
if [[ ! $eth_key =~ ^0x[0-9a-fA-F]{64}$ ]]; then
    echo "Invalid Ethereum private key in file. It should be 66 characters including '0x'."
    exit 1
fi

# Test AWS credentials
aws sts get-caller-identity > /dev/null
if [ $? -ne 0 ]; then
    echo "Error: AWS credentials not available or invalid"
    exit 1
fi

################################################################################
# Step 2: Convert private key to DER format
################################################################################

# Create a temporary working directory
work_dir=$(mktemp -d)
trap 'shred -u "$work_dir"/* && rm -rf "$work_dir"' EXIT

# Put PKCS/ASN.1 encoded key into eth_key_ec_params.der
key_hex=${eth_key:2}

ctx_secp256k1="302e0201010420"     # http://www.oid-info.com/get/1.3.132.0.10
ctx_bitstring="a00706052b8104000a"

echo $ctx_secp256k1 $key_hex $ctx_bitstring | xxd -r -p > "$work_dir/eth_key_ec_params.pkcs8"

# Convert the ASN1 encoded key params to PEM format
openssl ec \
    -inform DER -in "$work_dir/eth_key_ec_params.pkcs8" \
    -outform PEM -out "$work_dir/private_key.pem" 2>/dev/null

# Convert PEM to DER format
openssl pkcs8 -topk8 -nocrypt \
    -inform pem -in "$work_dir/private_key.pem" \
    -outform der -out "$work_dir/private_key.der"

# Check if the private key is valid
openssl ec -in "$work_dir/private_key.der" -check -noout
if [ $? -ne 0 ]; then
    echo "Error: Invalid private key"
    exit 1
fi

################################################################################
# Step 3: Get AWS KMS import parameters
################################################################################

# Get the KMS import parameters
aws kms get-parameters-for-import \
    --key-id "$aws_key_id" \
    --wrapping-algorithm RSAES_OAEP_SHA_256 \
    --wrapping-key-spec RSA_4096 \
    --output json > "$work_dir/import_params.json"

# Extract the import token and public key from the JSON
cat "$work_dir/import_params.json" | jq -r '.PublicKey'   > "$work_dir/import_public_key.b64"
cat "$work_dir/import_params.json" | jq -r '.ImportToken' > "$work_dir/import_token.b64"

# Convert the base64 encoded import token and public key to binary
openssl enc -d -base64 -A -in "$work_dir/import_token.b64"      -out "$work_dir/import_token.bin"
openssl enc -d -base64 -A -in "$work_dir/import_public_key.b64" -out "$work_dir/import_public_key.bin"

################################################################################
# Step 4: Encrypt the private key with the import public key
################################################################################

# Encrypt the key with the import public key
openssl pkeyutl \
    -encrypt \
    -pubin \
    -in "$work_dir/private_key.der" \
    -inkey "$work_dir/import_public_key.bin" \
    -keyform DER \
    -pkeyopt rsa_padding_mode:oaep \
    -pkeyopt rsa_oaep_md:sha256 \
    -pkeyopt rsa_mgf1_md:sha256 \
    -out "$work_dir/wrapped_private_key.der.enc"

################################################################################
# Step 5: Import the private key into AWS KMS
################################################################################

# Import the private key into AWS KMS
aws kms import-key-material \
    --key-id "$aws_key_id" \
    --import-token fileb: "**********"
    --encrypted-key-material fileb://$work_dir/wrapped_private_key.der.enc \
    --expiration-model KEY_MATERIAL_DOES_NOT_EXPIRE
_DOES_NOT_EXPIRE
