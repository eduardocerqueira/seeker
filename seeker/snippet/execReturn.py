#date: 2021-10-18T17:08:08Z
#url: https://api.github.com/gists/38a910503b9615da48e90434769aa2eb
#owner: https://api.github.com/users/YiLi225

### 1. Dynamic execution
codeString = '''a,b = 4,5; print(f"a = {a} and b = {b}"); print(f"a+b = {a+b}")'''
output = exec(codeString)
## print the output
print(f'** Is the return from exec() is None? {output == None} **')
