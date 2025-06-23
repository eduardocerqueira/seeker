#date: 2025-06-23T17:01:39Z
#url: https://api.github.com/gists/a001b1dc69d2addfd019c2465e69f920
#owner: https://api.github.com/users/Lenochxd

t9_mapping = {
    '2': 'abc',
    '3': 'def',
    '4': 'ghi',
    '5': 'jkl',
    '6': 'mno',
    '7': 'pqrs',
    '8': 'tuv',
    '9': 'wxyz',
    '0': ' '
}

input_string = input("Enter T9 input: ").split()

text = ""
for char in input_string:
    if char[0] in t9_mapping:
        text += t9_mapping[char[0]][len(char) - 1]  # Get the character based on the number of presses
    else:
        text += char

print()
print(text)
