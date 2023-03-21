#date: 2023-03-21T17:01:20Z
#url: https://api.github.com/gists/8d28cc36ce120fa857bb20eed62f00c2
#owner: https://api.github.com/users/erikbtx

# Programa para verificar a média da nota
media1 = (float(input('Digite sua 1º média: ')))
media2 = (float(input('Digite sua 2º média: ')))
media3 = media1 + media2
media_final = (int(media3 / 2))
print('Média final: ',media_final)
if media_final >= 7:
    print('Parabéns! Você foi aprovado.')
else:
    print('Poxa, que pena! Você foi reprovado.')