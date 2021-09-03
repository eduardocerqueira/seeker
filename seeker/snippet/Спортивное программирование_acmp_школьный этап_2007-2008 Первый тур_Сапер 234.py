#date: 2021-09-03T17:14:11Z
#url: https://api.github.com/gists/4594a46860bce1018353c5f615ebefc3
#owner: https://api.github.com/users/Maine558

def kolvo_min(x):
    if x not in Koor_min:
        kolvo = 0
        a = [x[0],x[0]+1,x[0]-1]
        b = [x[1],x[1]+1,x[1]-1]
        for i in range(len(a)):
            for j in range(len(b)):
                x_new = [a[i],b[j]]
                if x_new in Koor_min:
                    kolvo += 1
                else:
                    continue
        if kolvo == 0:
            return "."
        else:
            return kolvo
    else:
        return "*"


N,M,K = [int(s) for s in input().split()]

Koor_min = []

if K > M*N:
    print("Такого быть не может!")
else:
    for i in range(K):
        Koor_min.append([int(s) for s in input().split()])
        if Koor_min[i][0] > N:
            print("Вы вышли за границу, количество строк =",N)
            break
        if Koor_min[i][1] > M:
            print("Вы вышли границу, ширина карты =",M)

for i in range(1,N+1):
    for j in range(1,M+1):
        Koor = [i,j]
        print(kolvo_min(Koor),end="")
    print(" ")