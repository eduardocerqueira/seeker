#date: 2022-08-31T17:02:22Z
#url: https://api.github.com/gists/40f30e4851f6528719649c825d2e1e2d
#owner: https://api.github.com/users/vttrifonov

class Heap:
    h = []   
    p = {}
  
    @property
    def head(self):
        return self.h[0]
        
    @property
    def n(self):
        return len(self.h)

    def place(self, x, i, j = None):
        self.h[i] = x
        if j is None:
            if x not in self.p:
                self.p[x] = set()
        else:
            self.p[x].remove(j)
        self.p[x].add(i)


    def up(self, x, i):
        while i > 0:
            j = (i-1)//2
            v = self.h[j]
            if v >= x:
                break
            self.place(v, i, j)
            i = j
        return i

    def down(self, x, i):
        while True:
            j = 2*i+1
            if j >= self.n:
                break
            v = self.h[j]
            j1 = j+1
            if j1 >= self.n:
                if x > v:
                    break
            else:
                v1 = self.h[j1]
                if v < v1:
                    if x > v1:
                        break
                    v, j = v1, j1
                else:
                    if x > v:
                        break
            self.place(v, i, j)
            i = j
        return i

    def add(self, x):
        self.h.append(x)
        self.place(x, self.up(x, self.n-1))

    def remove(self, x):
        i = self.p[x]
        i = list(i)[0]
        self.p[x].remove(i)

        x = self.h.pop()
        if i == self.n:
            return
        self.place(x, self.down(x, self.up(x, i)), self.n)
