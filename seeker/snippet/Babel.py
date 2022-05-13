#date: 2022-05-13T16:58:35Z
#url: https://api.github.com/gists/6a82ceac4fef92ce7a69433499a8a13a
#owner: https://api.github.com/users/YamYoshi

#!/usr/bin/env python3
def rng():
    a = 1103515245
    c = 1
    m = 2**32
    rng.current = (a*rng.current + c) % m
    return rng.current

print("Insert Key:")
rng.current = int(input())

def pages():
    page = [' ','q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m',', ','. ']
    fullpage = []
    rng_page = [rng()for i in range(3200)]
    for i in rng_page:
        gen_page = page[(i % 29)]
        fullpage.append(gen_page)
    string_page = ''.join(fullpage)
    print(string_page)
pages()