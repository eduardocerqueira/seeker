#date: 2022-01-04T17:10:57Z
#url: https://api.github.com/gists/a6b302108be7e6e00baaedfb78a984a8
#owner: https://api.github.com/users/csobankesmarki

# List plugged in Yubikeys and get the serial from the list
ykman list

# Generate new FIDO2 resident key on the Yubikey plugged in (keep on plugged in, only) and saving the attestation certificate
# Without speficing the '-O user="..."' there is going to be a 32x ascii 0 filled into the user part and would overwrites exisiting keys without asking
# Complex user="..." part creates a 31 char long string adding random values to the end (max length is 31 as 32nd must be 0)
# Parameter -Z <cipher> can be different, valid values can be checked with 'ssh -Q cipher' command, default is aes256-ctr when omitting
ssh-keygen -t ed25519-sk -a 64 -O resident -O user="$(echo -n <user>@<FQDN>_$(date +'%Y%m%d')_$(uuidgen | tr -d '-') | cut -c 1-31)" -O write-attestation=id_ed25519_sk_<yubikey serial>_attest -f id_ed25519_sk_<yubikey serial> -C "<user>@<FQDN>-$(date +'%Y%m%d')-<yubikey serial>" -Z sha256-gcm@openssh.com

# Listing exiting FIDO2 credentials with ykman
ykman --device <yubikey serial> fido credentials list

# Output will be like:
# ssh: 3c757365723e403c4651444e3e5f32303232303130345f656239363662373400 openssh
# where the 2nd field is the encoded string of the user="..." field
# Double check the user value
echo -e $(ykman --device <yubikey serial> fido credentials list | cut -d' ' -f2 | sed -e 's/\(..\)/\\x\1/g')

# To change passphrase on the private key part (optional)
# Parameter -a specifies the bcrypt rounds, default is 16 when omitted
# Parameter -Z see above
ssh-keygen -p -a 64 -f id_ed25519_sk_<yubikey serial> -Z aes256-gcm@openssh.com

# Create an OpenSSL DER formatted certifcate file from the OpenSSH attestation certificate file
# File content can be checked with e.g. 'hexdump -C id_ed25519_sk_<yubikey serial>_attest' command
# OpenSSL certificates starts with a '30 82 <length in two bytes> 30 82 <length in two bytes> a0 03 02 01 02 ..."
# OpenSSH prefixes the DER formatted certificate with '00 00 00 "ssh-sk-attest-v00" 00 00 <cert length in two bytes>"
# This command causes some extra part on the end of the OpenSSL DER file (approx. 80 bytes) but the OpenSSL skips that
dd if=id_ed25519_sk_<yubikey serial>_attest bs=25 skip=1 status=none of=id_ed25519_sk_<yubikey serial>_attest.der

# Convert DER format to PEM (optional)
openssl x509 -in id_ed25519_sk_<yubikey serial>_attest.der -inform der -out id_ed25519_sk_<yubikey serial>_attest.pem -outform pem

# Print out X.509 certificate (for DER file must add '-inform der' to command line)
openssl x509 -in id_ed25519_sk_<yubikey serial>_attest.pem -noout -text
