#date: 2021-09-03T17:14:07Z
#url: https://api.github.com/gists/5501deca7bedf71ead187c6a0f8e31fe
#owner: https://api.github.com/users/Maine558

def F(i):
    for j in range(2,int(i**0.5)+1):
        if i%j==0:
            return False
    return True



a=30000
b= 45000
counter = 0
for i in range(a,b):
    if F(i) and (i*i) >= 1e9 and (i*i) <= 2e9:
        counter +=1
print(counter)