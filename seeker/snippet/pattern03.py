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


def spaces(n):
    for i in range(n):
        print(end=' ')


def star_row(n):
    for i in range(n):
        print('*', end=' ')

rows = int(input("Enter the number of rows:"))
blanks = rows
for i in range(rows+1):
    spaces(blanks)
    star_row(i)
    blanks -= 1
    print()