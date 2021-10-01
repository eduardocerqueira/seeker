#date: 2021-10-01T00:52:27Z
#url: https://api.github.com/gists/768bd546ae0d03bf79a17c33f0ad3b29
#owner: https://api.github.com/users/jgabriellima

import binascii
import StringIO
from Crypto.Cipher import AES


KEY = 'ce975de9294067470d1684442555767fcb007c5a3b89927714e449c3f66cb2a4'
IV = '9AAECFCF7E82ABB8118D8E567D42EE86'
PLAIN_TEXT = "ciao"

class PKCS7Padder(object):
    '''
    RFC 2315: PKCS#7 page 21
    Some content-encryption algorithms assume the
    input length is a multiple of k octets, where k > 1, and
    let the application define a method for handling inputs
    whose lengths are not a multiple of k octets. For such
    algorithms, the method shall be to pad the input at the
    trailing end with k - (l mod k) octets all having value k -
    (l mod k), where l is the length of the input. In other
    words, the input is padded at the trailing end with one of
    the following strings:

             01 -- if l mod k = k-1
            02 02 -- if l mod k = k-2
                        .
                        .
                        .
          k k ... k k -- if l mod k = 0

    The padding can be removed unambiguously since all input is
    padded and no padding string is a suffix of another. This
    padding method is well-defined if and only if k < 256;
    methods for larger k are an open issue for further study.
    '''
    def __init__(self, k=16):
        self.k = k

    ## @param text: The padded text for which the padding is to be removed.
    # @exception ValueError Raised when the input padding is missing or corrupt.
    def decode(self, text):
        '''
        Remove the PKCS#7 padding from a text string
        '''
        nl = len(text)
        val = int(binascii.hexlify(text[-1]), 16)
        if val > self.k:
            raise ValueError('Input is not padded or padding is corrupt')

        l = nl - val
        return text[:l]

    ## @param text: The text to encode.
    def encode(self, text):
        '''
        Pad an input string according to PKCS#7
        '''
        l = len(text)
        output = StringIO.StringIO()
        val = self.k - (l % self.k)
        for _ in xrange(val):
            output.write('%02x' % val)
        return text + binascii.unhexlify(output.getvalue())


def encrypt(my_key=KEY, my_iv=IV, my_plain_text=PLAIN_TEXT):
  """
    Expected result if called without parameters:

    PLAIN 'ciao'
    KEY 'ce975de9294067470d1684442555767fcb007c5a3b89927714e449c3f66cb2a4'
    IV '9aaecfcf7e82abb8118d8e567d42ee86'
    ENCRYPTED '62e6f521d533b26701f78864c541173d'
  """

  key = binascii.unhexlify(my_key)
  iv = binascii.unhexlify(my_iv)

  padder = PKCS7Padder()
  padded_text = padder.encode(my_plain_text)
  
  encryptor = AES.new(key, AES.MODE_CFB, iv, segment_size=128) #Initialize encryptor
  result = encryptor.encrypt(padded_text)

  return {
    "plain" : my_plain_text,
    "key": binascii.hexlify(key),
    "iv": binascii.hexlify(iv),
    "ciphertext": result
  }
  
if __name__ == '__main__':
  result = encrypt()
  print "PLAIN %r" % result['plain']
  print "KEY %r" % result['key']
  print "IV %r" % result['iv']
  print "ENCRYPTED %r" % binascii.hexlify(result['ciphertext'])