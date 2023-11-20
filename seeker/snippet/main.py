#date: 2023-11-20T16:31:02Z
#url: https://api.github.com/gists/31b41cb4c06acc4996755585ef921271
#owner: https://api.github.com/users/srezal

BASE = 256
PRIME_NUMBER = 9973


def hash_string(string: str, base: int, prime_number: int) -> int:
    """Function hashing string"""
    string_length = len(string)
    if string_length == 0:
        return 0
    expression = ord(string[string_length - 1]) * base**(string_length - 1)
    return expression  + hash_string(string[:-1:], base, prime_number) % prime_number


def find_substring(string: str, pattern: str) -> list:
    """Function findind substrings in string"""
    if pattern == "":
        return [0]
    pattern_length = len(pattern)
    base = BASE
    prime_number = PRIME_NUMBER
    pattern_hash_ = hash_string(pattern, base, prime_number)
    occurrences = []
    for i in range(len(string) - pattern_length + 1):
        if hash_string(string[i:i + pattern_length], base, prime_number) == pattern_hash_:
            if string[i:i + pattern_length] == pattern:
                occurrences.append(i)
    return occurrences
