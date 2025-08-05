#date: 2025-08-05T17:08:28Z
#url: https://api.github.com/gists/b7474568d9474624320b9cbb5d058d1d
#owner: https://api.github.com/users/nikolaydrumev

# a = [1, 2, 3,  4]
# b = splice(a, 1, 1)
# # a: [1, 3, 4]
# # b: [2]

# a = [1, 2, 3,  4]
# b = splice(а, 1, 2, 5, 6)
# # a: [1, 5, 6, 4]
# # b: [2, 3]

# a = [1, 2, 3,  4]
# b = splice(a, 1)
# # a: [1]
# # b: [2, 3, 4]

# a = [1, 2, 3,  4]
# b = splice(a, 2, 2, 7, 8, 9);
# # a: [1, 2, 7, 8, 9]
# # b: [3, 4]

class Splicer:
    def __init__(self, arr):
        self.arr = arr

    def splice(self, start, count, *args):
        length = len(self.arr)
        removed = []
        actual_count = 0
        i = start

        while i < length and actual_count < count:
            removed.append(self.arr[i])

            j = i
            while j < length - 1:
                self.arr[j] = self.arr[j + 1]
                j += 1

            self.arr.pop()
            length -= 1
            actual_count += 1

        if len(args) > 0:
            num_new = len(args)

            for _ in range(num_new):
                self.arr.append(0)

            length = len(self.arr)

            i = length - 1
            while i >= start + num_new:
                self.arr[i] = self.arr[i - num_new]
                i -= 1

            for k in range(num_new):
                self.arr[start + k] = args[k]

        return removed

a = [1, 2, 3, 4]
sp = Splicer(a)
b = sp.splice(1, 1)
print(sp.arr)
print(b)