#date: 2022-05-12T16:59:06Z
#url: https://api.github.com/gists/0d2730e63afc97609d4df79fe2b39981
#owner: https://api.github.com/users/anumukhe

class CryptoTrie:
    
    __lock = threading.RLock()
    __iv = Random.new().read(AES.block_size)
    
    def __init__(self):
        self.root = CryptoTrieNode()
        
        # For Generating AES Cipher text
        password = "SamPLeCrypticPassword"
        # add a "salt" to increase the entropy
        salt_bytes = 8
        key_bytes = 32
         
        salt = Random.new().read(salt_bytes)

        # Password-based key derivation function 2
        self.secret_key = PBKDF2(password, salt, key_bytes)