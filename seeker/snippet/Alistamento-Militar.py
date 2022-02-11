#date: 2022-02-11T16:41:26Z
#url: https://api.github.com/gists/3c90918a5ef775e51e07124c2db78f65
#owner: https://api.github.com/users/Jeanziin

from datetime import date
atual = date.today().year

nasc = int(input('Digite em qual ano você nasceu: '))
idade = abs(atual - nasc)
print('Quem Nasceu em {}, tem {} Anos em {}.'.format(nasc, idade, atual))


if idade == 18:
    print('Você tem que se alistar IMEDIATAMENTE! ')
elif idade > 18:
    total = abs(18 - idade)
    print('Você já deveria ter se alistado há {} anos'.format(total))
    alis = abs(atual - total)
    print('Seu Alistamento foi em {} '.format(alis))

elif idade < 18:
    total = abs(idade - 18)
    print('Ainda faltam {} Anos para seu alistamento'.format(total))
    alis = abs(atual + total)
    print('Seu Alistamento Será em {}'.format(alis))

print('''[ 1 ] HOMEM  [ 2 ] MULHER''')
a1 = int(input('Você é Homem ou Mulher ?'))
if a1 == 1:
        print('ALISTAMENTO OBRIGATÓRIO!')
elif a1 == 2:
        print('ALISTAMENTO NÃO OBRIGATÓRIO!')
else:
    print('NÚMERO INVÁLIDO!!!!')
