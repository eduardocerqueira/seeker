#date: 2022-12-20T16:57:47Z
#url: https://api.github.com/gists/a3cd2745fd12d90a449f0ac316a70ddc
#owner: https://api.github.com/users/miktorius

# in state list: 1 means occupied, 0 means empty and -1 means deleted
class Node:
    def __init__(self, key):
        self.key = key
        self.next = None


class DoubleHashing:
    def __init__(self, size=100, load_factor=0.75):
        self.items_count = 0
        self.load_factor = load_factor
        self.table = [None] * size
        self.state = [0] * size

    def hash_function(self, key, size=None):
        if not size: size = len(self.table)
        return key % size
    
    def __rehash(self):
        new_table = [None] * len(self.table) * 2
        new_state = [0] * len(self.table) * 2
        for bucket in self.table:
            if not bucket: continue
            self.__insert(bucket, new_table, new_state)
        return new_table, new_state

    def __insert(self, key, table=None, state=None):
        if not table: table = self.table
        if not state: state = self.state
        index, h = self.hash_function(key), 1
        hash_two = second_hash_function(key)
        while self.state[index] == 1:
            index = (index + h * hash_two) % len(self.table)
            h += 1
        table[index], state[index] = key, 1
    
    def insert(self, key):
        self.items_count += 1
        load_factor = self.items_count / len(self.table)
        if load_factor > self.load_factor:
            self.table, self.state = self.__rehash()
            self.load_factor = load_factor
        self.__insert(key)


    def search(self, key):
        index, h = self.hash_function(key), 1
        hash_two = second_hash_function(key)
        while (self.table[index] != key or\
            self.state[index] == -1) and\
                self.state[index] == 1:
            index = (index + h * hash_two) % len(self.table)
            h += 1
        if self.table[index] == key:
            return index
        return -1

    def delete(self, key):
        index = self.search(key)
        if index > -1:
            self.state[index] = -1