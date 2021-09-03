#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558


#задача: дано число, посчитать количество его чётных делителей

def count_div(n):
    counter=0
    oper=0
    for i in range(1,int(n**0.5)+1):
        oper+=1
        if(n%i==0):
            counter+=2 #прибавляем 2, т.к. i и n//i
    oper+=1
    if((n**0.5)%1==0):
        counter-=1
    return [counter,oper]

def count_even_div(n):
    counter=0
    i=1
    while(i*i<=n):
        if(n%i==0):
            if(i%2==0):
                counter+=1
            if(n//i % 2 == 0):
                counter+=1
        i+=1
    if(n%2==0)and(i*i==n):
        counter-=1
    return counter

def is_prime(n):
    for i in range(2,int(n**0.5)+1):
        if(n%i==0):
            return [False,i]
    return [True,int(n**0.5)]

def resheto(n):
    k=0
    a=[True]*(n+1) #презумпция простоты: пока не доказано,
                    #что число составное, мы считаем его простым
    a[0]=False #ноль - не простое число!
    a[1]=False #1 - не простое число
    for i in range(2,int(n**0.5)+1):
        k+=1
        if(a[i]): #перевод на русский: если i - простое
            if(i==2):
                j=2
                while (j * i <= n):
                    a[j * i] = False
                    j += 1
                    k += 1
            else:
                j=i
                while(j*i<=n):
                    a[j*i]=False
                    j+=2
                    k+=1
    print('Количество операций решето:',k)
    return a

#Задача: вывести все простые числа до n включительно
n=int(input())

#первая версия: через решето
a=resheto(n)
counter=0
for i in range(2,n):
    z=i
    if(a[i]):
        counter+=1
print(counter)

#вторая версия: через is_prime():
oper=0
counter=1
for i in range(3,n,2):
    a,b=is_prime(i)
    oper+=b
    if(a):
        counter+=1
print(counter)
print('Количество операций is_prime():',oper)