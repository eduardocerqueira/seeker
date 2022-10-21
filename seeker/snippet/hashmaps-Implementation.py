#date: 2022-10-21T17:11:10Z
#url: https://api.github.com/gists/dbaa5e7a1771acf0cdd9be02d4b813b2
#owner: https://api.github.com/users/a25osman

import unittest

class HashTable:
    """My custom implementation of hashtables in python using an associative array with chaining to handle
    collisions. Includes functionality to allow setting of key value pairs, getting of value with key, and
    deletion of key value pairs. Dynamically resizes as associative array's load factor passes threshold.
    """

    def __init__(self):
        self.size = 10
        self.hashmap = self.generate_associative_array()  # Associative array of n (size) number of buckets.
        self.number_key_value_pairs = 0
    
    def generate_associative_array(self):
        return [[] for _ in range(self.size)]

    def hash_fcn(self, key):
        """Hashing of key through python's built-in hash function and returning index for associative array."""
        hashed_key_index = hash(key) % self.size
        return hashed_key_index

    def is_load_factor_over_limit(self):
        """Load factor used to determine performance of hash table, primarily in relation to number of collisions.
        If Load Factor is greater than the threshold, then the associative array (self.hashmap) must be resized
        in order to ensure the performance of the hash table.
        """
        number_buckets = len(self.hashmap)
        load_factor = self.number_key_value_pairs / number_buckets
        LOAD_FACTOR_THRESHOLD = 0.75  # Value is from: Information and Software Technology (Owolabi, 2003) 
        return load_factor >= LOAD_FACTOR_THRESHOLD
    
    def resize_hashmap(self):
        """Dynamically resize hashmap by factor of 2. This will rehash all keys from existing hashmap
        into a new hashmap.
        """
        RESIZE_FACTOR = 2
        self.size = self.size * RESIZE_FACTOR
        self.number_key_value_pairs = 0
        new_hashmap = self.generate_associative_array()
        # Must go through every key in existing buckets and find its new corresponding hashed key index
        for bucket in self.hashmap:
            if bucket:
                for key_value_pair in bucket:
                    key, value = key_value_pair
                    hashed_key_index = self.hash_fcn(key)
                    new_bucket = new_hashmap[hashed_key_index]
                    self.set_value_helper(key, value, new_bucket)
        self.hashmap = new_hashmap

    def set_value_helper(self, key, value, bucket):
        """Helper function to set key value pair in hashmap. Used in functions: set_value and resize_hashmap."""
        key_exists = False
        for i, key_value_pair in enumerate(bucket):
            stored_key, stored_value = key_value_pair
            if stored_key == key:
                key_exists = True
                break
        if key_exists:
            # key already exists - update with new value
            bucket[i] = ((key, value))
        else:
            # Adding unique key-value pair.
            bucket.append((key, value))
            self.number_key_value_pairs +=1
    
    def set_value(self, key, value):
        """Store key value pair as a tuple in the associative array, and also resize hashmap if loadfactor
        is over threshold.
        """
        hashed_key_index = self.hash_fcn(key)
        bucket = self.hashmap[hashed_key_index]
        self.set_value_helper(key, value, bucket)
        if self.is_load_factor_over_limit():
            print("Resizing hashmap...")
            self.resize_hashmap()

    def get_value(self, key):
        """Retrieve value of associated key from hashmap; if it exists"""
        hashed_key_index = self.hash_fcn(key)
        bucket = self.hashmap[hashed_key_index]
        for key_value_pair in bucket:
            stored_key, stored_value = key_value_pair
            if stored_key == key:
                return stored_value
        raise Exception('Key does not exist in hashmap')

    def delete_value(self, key):
        """Remove key value pair from hashmap using key"""
        hashed_key_index = self.hash_fcn(key)
        bucket = self.hashmap[hashed_key_index]
        key_exists = False
        for i, key_value_pair in enumerate(bucket):
            stored_key, stored_value = key_value_pair
            if stored_key == key:
                key_exists = True
                break
        if key_exists:
            bucket.pop(i)
            self.number_key_value_pairs -= 1
        else:
            raise Exception('Key does not exist in hashmap')


class TestHashTable(unittest.TestCase):
    def setUp(self):
        self.my_hashmap = HashTable()
        self.my_hashmap.set_value("apple", 3)
        self.my_hashmap.set_value("car", 23500)
        self.my_hashmap.set_value("bike", 400)
        self.python_hashmap = {"apple": 3, "car": 5000, "bike": 400}

    def test_get_set_val(self):
        # Should confirm that 5000 == 5000
        self.my_hashmap.set_value("car", 5000)
        self.assertEqual(self.python_hashmap["car"], self.my_hashmap.get_value("car"))

    def test_delete_val(self):
        # Should confirm that 2 == 2
        self.my_hashmap.delete_value("bike")
        del self.python_hashmap["bike"]
        self.assertEqual(len(self.python_hashmap), self.my_hashmap.number_key_value_pairs)

if __name__ == '__main__':
    unittest.main()