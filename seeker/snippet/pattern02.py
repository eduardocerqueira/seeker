#date: 2021-09-20T17:02:13Z
#url: https://api.github.com/gists/30f259353ee35887ec3d819a6266cd37
#owner: https://api.github.com/users/Ultimateudhaya

'''
1
1 2
1 2 3
1 2 3 4
1 2 3 4 5
'''
def number_row(n):
    for i in range(n):
        print(i+1, end=' ')


rows = int(input("Enter the number of rows: "))
for i in range(1, rows+1):
    number_row(i)
    print()
