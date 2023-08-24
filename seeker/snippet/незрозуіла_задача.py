#date: 2023-08-24T16:54:38Z
#url: https://api.github.com/gists/197156e4716e44f59fb4eb584eec32a9
#owner: https://api.github.com/users/WKeen2010

from turtle import *

def f(x):
    for v in range(4):
        forward(x)
        left(90)

m = int(input('m = '))
n = int(input('n = '))
s = []
reserv = []
a = 0

Matrix = [[input("M["+str(i)+"]["+str(j)+"]=") for j in range(n)] for i in range(m)]

print('')
print('Matrix:')

for i in range(m):
    print(Matrix[i])

for i in range(m):
    for j in range(n):
        if int(Matrix[i][j]) == 1:
            s = s + [i, j]
            up()
            goto(j * 50, i * -50)
            down()
            begin_fill()
            f(50)
            end_fill()

up()
goto(0, 50)
down()

for w in range(2):
    forward(n * 50)
    right(90)
    forward(m * 50)
    right(90)

if s == [0, 0, 1, 1, 2, 2]:
    a = 1

else:
    while a == 0:
        if s == [0, 0, 1, 1, 2, 2]:
            a = 1

        if abs(s[0] - s[2]) < 2 and abs(s[1] - s[3]) < 2:
            reserv.append(s[0])
            reserv.append(s[1])
            del s[0], s[1]

            if len(s) / 2 == 1:
                a = 1

            else:
                a = 0

        else:

            if len(reserv) != 0:

                if abs(reserv[0] - s[2]) < 2 and abs(reserv[1] - s[3]) < 2:
                    del s[2]

                else:
                    a = 0

            if len(s) >= 5:
                if abs(s[0] - s[4]) < 2 and abs(s[1] - s[5]) < 2:
                    del s[0], s[1]

                    if len(s) / 2 == 1:
                        a = 1


                    else:
                        a = 0

                else:

                    if len(reserv) != 0:

                        if abs(reserv[0] - s[2]) < 2 and abs(reserv[1] - s[3]) < 2:
                            del s[2], s[3]

                        else:
                            a = 0

                    else:
                        a = 0

            else:
                a = len(s) // 2

print('')
print(a)

for i in range(999):
    left(1)