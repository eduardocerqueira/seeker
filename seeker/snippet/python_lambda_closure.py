#date: 2021-10-29T17:05:58Z
#url: https://api.github.com/gists/ce11ef1cac67bfa787c9e145824845ef
#owner: https://api.github.com/users/Kristof-Mattei

max = 10

def get_functions():
  lst = []
  for i in range(0, max):
    lst.append(lambda: print(i))
  return lst

def get_functions_yield():
  for i in range(0, max):
    yield lambda: print(i)

def get_functions_yield_separate_var():
  for i in range(0, max):
    x = i
    yield lambda: print(x)

def get_functions_yield_separate_context():
  for i in range(0, max):
    yield (lambda x: lambda: print(x))(i)

def for_f(iterator): 
  for f in iterator():
    f()

def for_f_preload(iterator): 
  for f in [x for x in iterator()]:
    f()


# functions:
functions = [
  get_functions,
  get_functions_yield,
  get_functions_yield_separate_var,
  get_functions_yield_separate_context,
]

# iterations
iterations = [
  for_f,
  for_f_preload
]

for f in functions:
  for i in iterations:
    print(f.__name__)
    print(i.__name__)

    i(f)