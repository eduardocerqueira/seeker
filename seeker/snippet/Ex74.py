#date: 2023-05-22T16:45:54Z
#url: https://api.github.com/gists/d94bb1fe8f38f0f18816fdb13a042e76
#owner: https://api.github.com/users/Dariosilv

from random import randint
numeros = (randint(1 ,10), randint(1 , 10), randint(1 , 10),
           randint(1  , 10), randint(1 , 10))
print('Os valores sorteados foram :', end='')
for n in numeros:
    print(f'{n}',end='')
print(f'\n O maior valor sorteado foi {max(numeros)}')
print(f'O menor valor sorteaddo foi {min(numeros)}')

