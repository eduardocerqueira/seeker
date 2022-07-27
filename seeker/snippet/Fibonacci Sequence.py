#date: 2022-07-27T17:06:31Z
#url: https://api.github.com/gists/fbb5fa414f52c96ed342c42c940922c9
#owner: https://api.github.com/users/DurjoyAcharya

class Fibonacci:
    def __init__(self) -> None:
        pass
    #Given code takes too many time to execute for recursive call in CoLab & Jupyter Notebook
    def Fibo_1(n: int) -> int:
        if n==0:
            return 0
        elif n==1:
            return 1
        else:
            return Fibonacci.Fibo_1(n-1) + Fibonacci.Fibo_1(n-2)
    #This function is used to find the nth Fibonacci number sequentially        
    def FibonacciSeries_V1(n: int):
        for i in range(n):
            print(Fibonacci.Fibo_1(i),end=' ')

    #Given iterative code takes faster than the above function to execute
    def Fibo_2(n:int)->int:
        fibo=[]
        fibo.append(0)
        fibo.append(1)
        for i in range(2,n+1):
            fibo.append(fibo[i-1]+fibo[i-2])
        return fibo[n]
    def FibonacciSeries_V2(n: int) -> list:
        for i in range(n):
            print(Fibonacci.Fibo_2(i),end=' ')
if __name__=='__main__':
    Fibonacci.FibonacciSeries_V1(50) #Work smoothly for small integers not for large integers
    print('--------------------------')
    Fibonacci.FibonacciSeries_V2(50) #Work smoothly for large integers not for small integers