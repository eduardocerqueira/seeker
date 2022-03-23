#date: 2022-03-23T17:03:39Z
#url: https://api.github.com/gists/020eac87bc9d2c6bc74534b0c9025de3
#owner: https://api.github.com/users/Sonegors

num = int(input('Digite um numero:'))
print('''Escolhas uma das bases para conversão:
[1] converter para BINÁRIO
[2] converter para OCTAL
[3] converter para HEXADECIMAL''')
opção = int(input('Sua opção:'))
if opção ==1:
    print('{} converter para BINÁRIO é igual a {}'.format(num, bin(num)[2:]))
elif opção == 2:
    print('{} convertdo para OCTAL é igaul a {}'.format(num, oct(num)[2:]))
elif opção == 3:
    print('{} converdito para HEXADECIMAL é igual a {}'.format(num, hex(num)[2:]))
else:
    print('Opção invalida tente novamente!')