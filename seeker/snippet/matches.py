#date: 2022-10-19T17:27:21Z
#url: https://api.github.com/gists/aeef1a7ee10e1ed97ec46df0f2a841f6
#owner: https://api.github.com/users/AnasImloul

def mask(number, bits):
    return number & ((1 << bits) - 1)


def matches(number, target, bits):
    # return the number of bits that match
    return bin(mask(number, bits) ^ mask(target, bits))[2:].zfill(bits).count('0')


def intify(string):
    if string[0] == "m":
        return int(string.replace("m", "1").replace("h", "0"), 2)
    else:
        return int(string.replace("h", "1").replace("m", "0"), 2)


def solve(string):
    original = intify(string)
    current = original >> 1
    move = 1
    score = 0,0
    for bits in range(len(string)-1, 0, -1):
        temp_score = matches(current, original, bits)
        if temp_score > score[0]:
            score = temp_score, move
        current >>= 1
        move += 1
    return score

print(*solve(input()))