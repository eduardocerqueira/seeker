#date: 2021-09-20T17:02:13Z
#url: https://api.github.com/gists/30f259353ee35887ec3d819a6266cd37
#owner: https://api.github.com/users/Ultimateudhaya

'''
*
* *
* * *
* * * *
* * * * *
* * * * * *
'''


def star_row(n):
    for i in range(n):
        print('*', end=' ')


rows = int(input("Enter the number of rows: "))
for i in range(1, rows+1):
    star_row(i)
    print()
