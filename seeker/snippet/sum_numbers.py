#date: 2023-01-26T16:56:36Z
#url: https://api.github.com/gists/3d961548d5779dab566a26676ddec4a3
#owner: https://api.github.com/users/Rockaroller

random_numbers = input('Enter a number:')
print('\n')
num = random_numbers.split()
print(num)

for x in range(len(num)):
    num[x] = int(num[x])

print(sum(num))
