#date: 2025-11-13T16:56:25Z
#url: https://api.github.com/gists/bfcccc0ad099f76dbf6f067acb64b77a
#owner: https://api.github.com/users/Mdchan786

import base64


class StringFogImpl:
    CHARSET_NAME_UTF_8 = "UTF-8"

    @staticmethod
    def decrypt(string):
        return StringFogImpl().decrypt(string, StringFogImpl.CHARSET_NAME_UTF_8)

    def encrypt(self, string, key):
        try:
            return base64.b64encode(self.xor(string.encode(StringFogImpl.CHARSET_NAME_UTF_8), key)).decode()
        except UnicodeEncodeError:
            return base64.b64encode(self.xor(string.encode(), key)).decode()

    def decrypt(self, string, key):
        try:
            return self.xor(base64.b64decode(string), key).decode(StringFogImpl.CHARSET_NAME_UTF_8)
        except UnicodeDecodeError:
            return self.xor(base64.b64decode(string), key).decode()

    @staticmethod
    def overflow(string, key):
        return string is not None and (len(string) * 4) / 3 >= 65535

    @staticmethod
    def xor(byte_array, key):
        length = len(byte_array)
        key_length = len(key)
        result = bytearray(length)
        for i in range(length):
            result[i] = byte_array[i] ^ ord(key[i % key_length])
        return bytes(result)


string_fog = StringFogImpl()

encrypted_string = string_fog.encrypt("Hello, World!", "encryption_key")
print("Encrypted:", encrypted_string)

decrypted_string = string_fog.decrypt(encrypted_string, "encryption_key")
print("Decrypted:", decrypted_string)


decrypted_string = string_fog.decrypt(encrypted_string, "UTF-8")
print("Decrypted:", decrypted_string)
