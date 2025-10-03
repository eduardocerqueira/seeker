#date: 2025-10-03T17:03:43Z
#url: https://api.github.com/gists/f28c16eac65bbc368ce5612de5e3ed05
#owner: https://api.github.com/users/xtornasol512

class Base62Tools:
    """ 
    Utils to manage Base62 encode and decode
    Base on this snippet https://gist.github.com/agharbeia/5d3dc0e6998c0f1458dd9b35e57118db
    """
    digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    # TODO enhance removing example = "lL10oO"
    # digits = '23456789ABCDEFGHJKMNPQRSTWXYZabcdefghjkmnpqrstwxyz'
    # See more details in Crockford's Base32
    radix = len(digits)

    @staticmethod
    def encode(number):
        """ Encode a number in base10 to string Base62 """
        ZERO = 0
        if not isinstance(number, int):
            raise Exception("[Base62Tools ERROR] Not number received: {}".format(number))
        if number < ZERO:
            raise Exception("[Base62Tools ERROR] Negative number {} received".format(number))
        if number == ZERO:
            return str(ZERO)
        result = ''
        while number != ZERO:
            result = (Base62Tools.digits[number % Base62Tools.radix]) + result
            number = int(number / Base62Tools.radix)
        return result

    @staticmethod
    def decode(base62_string):
        """ Decode a string in base62 to a number in base10 """

        base62_string = str(base62_string)

        if isinstance(base62_string, int):
            base62_string = str(base62_string)

        if not isinstance(base62_string, str):
            raise Exception("[Base62Tools ERROR] Not string received: {}".format(base62_string))

        number_result, exp = 0, 1
        for character in reversed(base62_string):
            number_result += exp * Base62Tools.digits.index(character)
            exp *= Base62Tools.radix
        return number_result
