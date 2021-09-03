#date: 2021-09-03T17:13:37Z
#url: https://api.github.com/gists/f82708142ebb84d0e3b6b76456fe6047
#owner: https://api.github.com/users/Maine558

def high_and_low(numbers):
    a = [int(s) for s in numbers.split(" ")]
    numbers = ""
    numbers += str(max(a)) + " " + str(min(a))
    return numbers

print(high_and_low("4 5 29 54 4 0 -214 542 -64 1 -3 6 -6"))