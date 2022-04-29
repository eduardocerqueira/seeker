#date: 2022-04-29T16:48:02Z
#url: https://api.github.com/gists/a2dc4b9ab672250201341f077d2053b5
#owner: https://api.github.com/users/PMExtra

def solution(s):
    if not s: return 0

    result = 1
    min_divisor = 2
    while True:
        length = len(s)
        factor = 1
        # loop in a preprocessed prime number list can improve performance
        for n in range(min_divisor, length + 1):
            if length % n: continue
            chunk_size = length // n
            chunks = [s[i:i + chunk_size] for i in range(0, length, chunk_size)]
            if chunks.count(chunks[0]) != len(chunks): continue
            s = chunks[0]
            min_divisor = n
            factor = n
            break
        if factor == 1: break
        result *= factor

    return result