#date: 2022-01-10T17:05:09Z
#url: https://api.github.com/gists/58fbc6cd0b79528111b7eb37836e609c
#owner: https://api.github.com/users/rimuru72

#Code to print divisors for a dividend inputted by user

dividend = int(input("Please enter the number/dividend: "))

divisors = range(1, dividend+1)

for el in divisors:
    if dividend % el == 0:
        print(el)