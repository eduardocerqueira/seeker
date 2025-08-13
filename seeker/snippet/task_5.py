#date: 2025-08-13T16:40:41Z
#url: https://api.github.com/gists/c16e305a437ee7787571bd8dbb39ee8c
#owner: https://api.github.com/users/DanielIvanov19

class MyList(list):
    def __getitem__(self, index):
        # If index is a slice -> returning generator
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return (self[i] for i in range(start, stop, step))
        # If it's regular index -> standard behavior
        return super().__getitem__(index)