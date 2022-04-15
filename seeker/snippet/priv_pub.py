#date: 2022-04-15T16:57:19Z
#url: https://api.github.com/gists/8ddc6c7e021e2205b4723d2cf866e9ea
#owner: https://api.github.com/users/shrimalmadhur

    # Reference - https://stackoverflow.com/a/53488466 
    # pre req with python3
    # pip3 install ecdsa
    # pip3 install pysha3 # sha3 won't work
    
    from ecdsa import SigningKey, SECP256k1
    import sha3, random, binascii

    private_key = "<some private key in hex string>"
    private_key = bytes(private_key, 'utf-8')
    private_key = binascii.unhexlify(private_key)
    priv = SigningKey.from_string(private_key, curve=SECP256k1)
    pub = priv.get_verifying_key().to_string()
    keccak = sha3.keccak_256()
    keccak.update(pub)
    address = keccak.hexdigest()[24:]
    print(address, priv.to_string().hex())