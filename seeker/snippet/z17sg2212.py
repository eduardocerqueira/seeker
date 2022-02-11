#date: 2022-02-11T17:09:29Z
#url: https://api.github.com/gists/ae8d49fb00b41467d3f24525b98a53dd
#owner: https://api.github.com/users/alexei-math

f = open('17.txt', 'r')
# numbers = [2, 5, 9, 8, 10] ### test numbers
numbers = list(map(int, f.readlines()))

sum_max = 0
pair_count = 0

for idx in range(len(numbers) - 1):
    s = numbers[idx] + numbers[idx+1]
    if ( numbers[idx] % 5 == 0 or numbers[idx+1] % 5 == 0) and s % 7 == 0:
        pair_count += 1
        if s > sum_max:
            sum_max = s

print(pair_count, sum_max)

f.close()
