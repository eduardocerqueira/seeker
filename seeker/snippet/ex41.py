#date: 2024-05-30T17:06:41Z
#url: https://api.github.com/gists/d8c6376ec772741ee7900bb0b700f58f
#owner: https://api.github.com/users/motyzk

def funny(s):
    l = s.split(" ")
    result = []
    for w in l:
        result.append(w[::-1].title())
    return ' '.join(result)


def funny(s):
    result = [w[::-1].title() for w in s.split(" ")]
    return ' '.join(result)


def funny(s):
    return ' '.join([w[::-1].title() for w in s.split(" ")])


result = funny("Foo bar")
print(result)
assert result == "Oof Rab"

result = funny("The quick brown fox")
print(result)
assert result == "Eht Kciuq Nworb Xof"

print("OK")
