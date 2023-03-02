#date: 2023-03-02T17:08:26Z
#url: https://api.github.com/gists/aa3733727fb17fdf501d0482d8dd5c61
#owner: https://api.github.com/users/TheMatt2

"""
Simply little utility function to make it possible to pickle generators.
While generator instances can not actually be pickled, this saves the generator
function and only creates a generator instance when the first value is accessed.

The goal is to make it possible to pass generators, with arguments, through multiprocessing
which uses pickling internally to move objects between Python processes.

import multiprocessing

@pickle_generator
def gen_squares(a, b):
    for n in range(a, b):
        yield n * n

def worker_run(iterable):
    for i in iterable:
        print(i)

if __name__ == "__main__":
    # Because "fork" doesn't necessarily use pickling 
    multiprocessing.set_start_method("spawn")
    
    p = multiprocessing.Process(target = worker_run, args = (gen_squares(1, 9),))
    p.start()
    p.join()
"""
from functools import wraps

__author__ = "Matthew Schweiss"

class PickleGenerator:
    """
    Internal class to represent a wrapped generator.
    """
    def __init__(self, generator, *args, **kwargs):
        self.generator = generator
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter(self.generator(*self.args, **self.kwargs))

def pickle_generator(generator):
    @wraps(generator)
    def wrapper(*args, **kwargs):
        #return PickleGenerator(generator, *args, **kwargs)
        return PickleGenerator(generator, *args, **kwargs)

    generator.__qualname__ = f"{generator.__qualname__}.__wrapped__"
    return wrapper

@pickle_generator
def gen_doe():
   yield "doe"
   yield "ray"
   yield "me"

if __name__ == "__main__":
    print("Using generator directly:")
    for num in gen_doe():
        print(num)

    import pickle

    # Pickle the generator function
    pkl_gen_due = pickle.dumps(gen_doe)

    # Create instance from pickled generator
    gen_doe_inst = pickle.loads(pkl_gen_due)()

    # Pickle the generator instance
    pkl_test_gen_inst = pickle.dumps(gen_doe_inst)

    # Unpickle the generator instance and use
    print("Using pickled generator:")
    for num in pickle.loads(pkl_test_gen_inst):
        print(num)
