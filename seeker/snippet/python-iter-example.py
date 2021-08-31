#date: 2021-08-31T13:12:25Z
#url: https://api.github.com/gists/01d2865cab197d4b15d78baa6bb379d0
#owner: https://api.github.com/users/gilwo

class A:
    def __init__(self, N=500, K=40, L=3):
        self.chunk = L
        lst = [x for x in range(1, N+1)]
        self.z = [lst[i:i + K] for i in range(0, len(lst), K)]
        self.lst = lst
    
    def __iter__(self):
        x = []
        for batch in self.z:
            x += batch
            while len(x) > self.chunk:
                yield x[:self.chunk]
                x = x[self.chunk:]
        yield x
            
    def all(self):
        return self.z
            
a = A(11, 4, 5)

print(a.all())

# expected output:
#[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]]

for i in a:
    print(i)

# expected output:
#[1, 2, 3, 4, 5]
#[6, 7, 8, 9, 10]
#[11]
