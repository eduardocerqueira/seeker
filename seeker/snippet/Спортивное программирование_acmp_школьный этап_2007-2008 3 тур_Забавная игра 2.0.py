#date: 2021-09-03T17:14:11Z
#url: https://api.github.com/gists/4594a46860bce1018353c5f615ebefc3
#owner: https://api.github.com/users/Maine558

def new_binar(x):

    new_sibvol = ""

    for i in range(len(x)):

        if i != len(x) - 1:

            new_sibvol += x[i + 1]

        else:

            new_sibvol += x[0]

    return new_sibvol


N = int(input())

binar = bin(N)[2:]

a = []

n_b = new_binar(binar)

while n_b != binar:

    a.append(int(new_binar(n_b)))

    n_b = new_binar(n_b)



a.sort()

string_om = str(max(a))

ans = 0

for i in range(len(string_om)):

    if string_om[i] == "1":

        ans += 2 ** (len(string_om) - i - 1)

print(ans)

