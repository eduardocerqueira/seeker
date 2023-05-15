#date: 2023-05-15T17:04:27Z
#url: https://api.github.com/gists/dd743f6e7490a9ee9864de1904952629
#owner: https://api.github.com/users/yqNLP

class DSU:
    """
    并查集
    """
    def __init__(self, size):
        self.root = [i for i in range(size)]

    def find(self, x):
        if self.root[x] == x:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return 
        self.root[root_a] = root_b
        
if __name__ == '__main__':
    dsu = DSU(11)
    # 1, 3, 5 
    # 2, 4, 6
    # 7, 8, 9, 10
    dsu.union(1, 3)    
    dsu.union(3, 5)
    dsu.union(2, 4)
    dsu.union(4, 6)
    dsu.union(7, 9)
    dsu.union(8, 10)
    dsu.union(7, 10)
    for i in range(11):
        print(dsu.find(i))
