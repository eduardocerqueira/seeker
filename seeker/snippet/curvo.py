#date: 2022-12-15T16:56:05Z
#url: https://api.github.com/gists/9d4772cd259f9e481f1d03d30dcae729
#owner: https://api.github.com/users/SnowLew


# n dados
import random


n = int(input())
dados = {}

# Idade, genero (M,F), estado_civil (casado, solteiro), canal de tv(2 até 22), horario (0 até 23)
for i in range(n):
    idade = random.randint(18, 60)
    genero = random.choice(['M', 'F'])
    estado_civil = random.choice(['casado', 'solteiro'])
    canal = random.randint(2, 22)
    horario = random.randint(0, 23) 
    # usar dicionario
    dados[i] = {
        'idade': idade,
        'genero': genero,
        'estado_civil': estado_civil,
        'canal': canal,
        'horario': horario
    }
    print(idade, genero, estado_civil, canal, horario)

# 1. Qual o canal mais assistido? entre 12 e 18 horas

#array dos canais
canais = []
for i in range(2, 23):
    canais.append(0)


for i in range(n):
    if dados[i]['horario'] >= 12 and dados[i]['horario'] <= 18:
        canais[dados[i]['canal']-2] += 1

# canal mais assistido
maior = 0
for i in range(2, 23):
    if canais[i-2] > maior:
        maior = canais[i-2]
        canal = i

print('O canal mais assistido entre 12 e 18 horas é o', canal)

# canais vistos por solteiros e M
canais = []
for i in range(2, 23):
    canais.append(0)

for i in range(n):
    if dados[i]['estado_civil'] == 'solteiro' and dados[i]['genero'] == 'M':
        canais[dados[i]['canal']-2] += 1

# canal != 0
for i in range(2, 23):
    if canais[i-2] != 0:
        print('O canal', i, 'foi visto por', canais[i-2], 'solteiros e homens')

# porcetagem que assistiram canal 8
canal8 = 0
for i in range(n):
    if dados[i]['canal'] == 8:
        canal8 += 1

print('A porcentagem de pessoas que assistiram o canal 8 é', canal8/n*100, '%')

# a media da idade
media = 0
for i in range(n):
    media += dados[i]['idade']

print('A media da idade é', media/n)