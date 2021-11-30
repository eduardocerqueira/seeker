#date: 2021-11-30T16:59:04Z
#url: https://api.github.com/gists/d54cebe25a3a8dec7eafce63f9636844
#owner: https://api.github.com/users/juanfal

# t07e16.pivote.py
# juanfc 2021-11-30
# 

def posx(l, x):
    i = 0
    while i < len(l) and l[i] != x:
        i += 1
    if i < len(l):
        return i
    else:
        return -1




def pivote(l, x):
    px = posx(l, x)
    if px >= 0: # encontrado
        return l[:px], l[px:]
    else:
        return l

# ------------------

l = [1, 2, 3, 4, 5]

print(f"pivote: {3} -> {pivote(l, 3)}")
print(f"pivote: {1} -> {pivote(l, 1)}")
print(f"pivote: {5} -> {pivote(l, 5)}")
print(f"pivote: {-33} -> {pivote(l, -33)}")
