#date: 2022-10-11T17:18:32Z
#url: https://api.github.com/gists/5d87c1d76a75ad68557f729f27707b78
#owner: https://api.github.com/users/jeremy9959

class Gaussian:
    def __init__(self,a,b):
        self._real=a
        self._imag=b

    def real(self):
        return self._real

    def imag(self):
        return self._imag
    
    def norm(self):
        return self._real**2+self._imag**2

    def add(self,x):
        realpart = self._real+x.real()
        imagpart=self._imag+x.imag()
        return Gaussian(realpart,imagpart)

    def neg(self):
        realpart=-self._real
        imagpart=-self._imag
        return Gaussian(realpart,imagpart)

    def times(self,x):
        realpart=(self._real)*(x.real())-(self._imag)*(x.imag())
        imagpart=(self._real)*(x.imag())+(self._imag)*(x.real())
        return Gaussian(realpart,imagpart)

    def __repr__(self):
        return repr([self._real,self._imag])

    def bar(self):
        return Gaussian(self._real,-self._imag)
    
    def is_zero(self):
        if (self._real==0) and (self._imag==0):
            return True
        else:
            return False
    

def gauss_euclid(x,y):
    a,b=x,y
    while True:
        num=a.times(b.bar())
        denom=b.norm()
        q=Gaussian(round(num.real()/denom),round(num.imag()/denom))
        r=a.add((q.times(b)).neg())
        if r.is_zero():
            return b
        else:
            a,b=b,r