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


def spaces(n):
    for i in range(n):
        print(end=' ')


def number_row(n):
    for i in range(n):
        print(i+1, end=' ')


rows = int(input("Enter the number of rows:"))
blanks = rows
for i in range(rows+1):
    spaces(blanks)
    number_row(i)
    blanks -= 1
    print()