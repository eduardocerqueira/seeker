#date: 2023-01-25T16:54:16Z
#url: https://api.github.com/gists/5d61896e14a70b9ab4db1d9f23cad48a
#owner: https://api.github.com/users/mBohunickaCharles

s = "singing in the rain and playing in the rain are two entirely different situations but both can be fun"
vowels = ['a','e','i','o','u']

num_vowels = 0

for i in s:
    for v in vowels:
        if v == i:
            num_vowels += 1
            
print(num_vowels)