#date: 2024-05-30T17:08:16Z
#url: https://api.github.com/gists/e6e3731f47d48b413d18fa0283fb1e19
#owner: https://api.github.com/users/motyzk

countries = ['Israel', 'Jordan', 'Egypt']
expected = ['IS', 'JO', 'EG']

result = []
for c in countries:
    result.append(c[:2].upper())

result = [c[:2].upper() for c in countries]

print(result)
assert result == expected



countries = ['Israel', 'Jordan', 'Egypt']
expected = ['IS', 'EG']

result = []
for c in countries:
    if c[0] in 'AEIOU':
        result.append(c[:2].upper())

result = [c[:2].upper() for c in countries if c[0] in 'AEIOU']

print(result)
assert result == expected
