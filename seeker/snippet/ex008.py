#date: 2021-12-16T17:08:43Z
#url: https://api.github.com/gists/32c0d66626d516797363677812cb12cf
#owner: https://api.github.com/users/LeandroCostaSantos

medida = int(input('Digite a medida (em metros): '))
if medida == 0:
    print('Esta é uma medida inválida.', end='')
if medida == 1:
    print(f'Isto tem {medida} metro, ou', end=' ')
    print(f'{medida * 100} centímetros, ou', end=' ')
    print(f'{medida * 1000} milímetros.')
elif medida >= 2:
    print(f'Isto tem {medida} metros, ou', end=' ')
    print(f'{medida*100} centímetros, ou', end=' ')
    print(f'{medida*1000} milímetros.')
