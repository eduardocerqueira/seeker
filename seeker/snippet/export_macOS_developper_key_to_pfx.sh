#date: 2024-05-30T16:57:28Z
#url: https://api.github.com/gists/8881e0550e66ddda63d6b54cb4d3d2d8
#owner: https://api.github.com/users/gllmAR

#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: "**********"
    exit 1
fi

# Set the export password from the arguments
EXPORT_PASSWORD= "**********"

# Find the code signing certificate (Apple Development or Developer ID Application)
CERT_SHA1=$(security find-identity -p codesigning -v | grep -E 'Developer ID Application|Apple Development' | awk '{print $2}' | head -n 1)

if [ -z "$CERT_SHA1" ]; then
    echo "Code signing certificate not found in the keychain"
    exit 1
fi

echo "Using certificate with SHA-1: $CERT_SHA1"

# Export the private key and certificate from the keychain to a P12 file
security export -k ~/Library/Keychains/login.keychain-db -t identities -f pkcs12 -P "$EXPORT_PASSWORD" -o temp.p12
if [ $? -ne 0 ]; then
    echo "Failed to export the P12 file"
    exit 1
fi

echo "...exported P12 file successfully"

# Convert the P12 file to PEM format using legacy algorithms
openssl pkcs12 -in temp.p12 -out temp.pem -nodes -legacy -passin pass: "**********"
if [ $? -ne 0 ]; then
    echo "Failed to convert the P12 file to PEM format"
    rm temp.p12
    exit 1
fi

echo "...converted P12 to PEM format successfully"

# Extract the private key and certificate from the PEM file
openssl pkey -in temp.pem -out private_key.pem
if [ $? -ne 0 ]; then
    echo "Failed to extract the private key from the PEM file"
    rm temp.p12 temp.pem
    exit 1
fi

openssl x509 -in temp.pem -out certificate.pem
if [ $? -ne 0 ]; then
    echo "Failed to extract the certificate from the PEM file"
    rm temp.p12 temp.pem private_key.pem
    exit 1
fi

echo "...extracted private key and certificate successfully"

# Combine the private key and certificate into a PFX file
openssl pkcs12 -export -inkey private_key.pem -in certificate.pem -out cert.pfx -password pass: "**********"
if [ $? -ne 0 ]; then
    echo "Failed to create the PFX file"
    rm temp.p12 temp.pem private_key.pem certificate.pem
    exit 1
fi

echo "...created PFX file successfully"

# Check the PFX file content
if [ ! -s cert.pfx ]; then
    echo "PFX file is empty or does not exist"
    rm temp.p12 temp.pem private_key.pem certificate.pem cert.pfx
    exit 1
fi

# Display file size for debugging
echo "PFX file size: $(stat -f%z cert.pfx) bytes"

# Encode the PFX file in base64 format
base64 -i cert.pfx -o cert.pfx.base64
if [ $? -ne 0 ]; then
    echo "Failed to encode the PFX file in base64 format"
    rm temp.p12 temp.pem private_key.pem certificate.pem cert.pfx
    exit 1
fi

echo "...encoded PFX file in base64 format successfully"

# Clean up temporary files
rm temp.p12 temp.pem private_key.pem certificate.pem cert.pfx

echo "The PFX file has been encoded in base64 and saved as cert.pfx.base64"
x.base64"
