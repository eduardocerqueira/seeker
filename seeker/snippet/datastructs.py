#date: 2023-07-27T16:54:50Z
#url: https://api.github.com/gists/fe2058f22c6296961e15d80bf38a9dd5
#owner: https://api.github.com/users/SightSpirit

'''
Description:
I've always had trouble wrapping my head around some data structures, so I decided to practice implementing the easier ones in Python. This module is the result.

Example Use Case:
So many possibilities! These serve as alternatives to the `list` structure.

License:
This code is in the public domain. You may use it for any purpose that is considered legal in your jurisdiction, including for-profit purposes, without providing attribution or including a similar license. You may attribute to Eden Biskin, if desired.
'''

class AbstractOrderedStructure():
    
    def __init__(self, data=None):
        self._list = list(x for x in data) if data is not None else []

    def push(self, datum):
        self._list.append(datum)

    def __str__(self):
        return str(self._list)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._list)

    __hash__ = None

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        i = self._index
        if self._index < len(self._list):
            self._index += 1
            return self._list[i]
        else:
            raise StopIteration()

    def __getitem__(self,i):
        return self._list[i]

    def __eq__(self, other):
        if hasattr(other, '_list'):
            return self._list == other._list
        elif type(other) is list:
            return self._list == other
        elif type(other) is tuple:
            if len(self._list) != len(other):
                return False
            else:
                for i,val in enumerate(self._list):
                    if val != other[i]:
                        return False
            return True
        else:
            raise TypeError(f'Cannot compare {type(self).__name__} and {type(other).__name__}.')


class Stack(AbstractOrderedStructure):

    def __init__(self, data=None):
        super().__init__(data)

    def pop(self):
        try:
            return self._list.pop()
        except IndexError as e:
            print(f'IndexError: pop from empty Stack')
            

class Queue(AbstractOrderedStructure):

    def __init__(self, data=None):
        super().__init__(data)

    def dequeue(self):
        if len(self._list) < 1:
            raise IndexError('Queue is empty')
        item = self._list[0]
        self._list = self._list[1:]
        return item

class Deque(Queue):

    def __init__(self, data=None):
        super().__init__(data)
        self.LEFT = 0
        self.RIGHT = 1

    def dequeue_l(self):
        return super().dequeue()

    def dequeue_r(self):
        if len(self._list) < 1:
            raise IndexError('Queue is empty')
        item = self._list[-1]
        self._list = self._list[:-1]
        return item

    def dequeue(self, direction):
        if type(direction) is str:
            if direction.lower().startswith('r'):
                direction = self.RIGHT
            elif direction.lower().startswith('l'):
                direction = self.LEFT
        if direction not in (self.LEFT, self.RIGHT):
            if type(direction) is int:
                raise ValueError('Must specify either right or left')
            else:
                raise TypeError('Must specify either right or left')

        if direction == self.LEFT:
            return self.dequeue_l()
        else:
            return self.dequeue_r()