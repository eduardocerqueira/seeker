#date: 2022-07-18T17:20:47Z
#url: https://api.github.com/gists/b5a7c815275aa3cb03a15ed1eb39dde8
#owner: https://api.github.com/users/g105b

#!/bin/bash

message="This is my message, I hope you can see it. It's very long now."
key="sup3r_s3cr3t_p455w0rd"

encrypted=$(
	echo -n "$message" | openssl enc \
	-aes-256-ctr \
	-e \
	-k "$key" \
	-iv "504914019097319c9731fc639abaa6ec"
)

echo -n $encrypted | xxd -p | tr -d "\n"
echo ""

decrypted=$(
	echo -n $encrypted | openssl enc \
	-aes-256-ctr \
	-d \
	-k "$key" \
	-iv "504914019097319c9731fc639abaa6ec"
)

echo $decrypted

### Output:
# *** WARNING : deprecated key derivation used.
# Using -iter or -pbkdf2 would be better.
# /home/g105b/Code/g105b/cipher-test/encrypt-decrypt.bash: line 12: warning: command substitution: ignored null byte in input
# 53616c7465645f5f9308d5d03474b132b76f6c7cacd3a3bd54627c8f0fd890578ed1c53fb91f8e4b3e3250d48337ec6baaee1eae1feb8d1243de5976623b43b9d55e52203aa030f46dfa34be527371
# *** WARNING : deprecated key derivation used.
# Using -iter or -pbkdf2 would be better.
# bad decrypt
# 4017C07C697F0000:error:1C80006B:Provider routines:ossl_cipher_generic_block_final:wrong final block length:../providers/implementations/ciphers/ciphercommon.c:429:
# This is my message, I hope you c�T�86��zĪe��9��

