#date: 2021-10-19T16:51:24Z
#url: https://api.github.com/gists/7af502720d9e99af576e002ce8dfba9b
#owner: https://api.github.com/users/donovan680

'''
Algoritmos gulosos
- Sempre escolhe a alternativa que parece mais promissora naquele instante
- NUNCA reconsidera essa decisão
- Uma escolha que foi feita NUNCA é revista
- Não há backtracking
- A escolha é feita de acordo com um criterio guloso - decisão localmente ótima.
- Nem sempre dão soluções ótimas

Problema do Troco (Troco mínimo)
Moedas = {100, 50, 25, 5, 1}
Troco: 75
Quantidade mínima de moedas: 2 (1 de 50 + 1 de 25)
'''

moedas = [100, 50, 25, 5, 1]
total = 0
troco = 130

for i in range(len(moedas)):
    num_moedas = troco // moedas[i]
    troco -= num_moedas * moedas[i]
    total += num_moedas

print(total)