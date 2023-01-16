#date: 2023-01-16T16:52:55Z
#url: https://api.github.com/gists/5ca3c944867470f81f401da6c5ef3e33
#owner: https://api.github.com/users/vitorguidi




# cada elemento de transaction vai ser  da forma
#  {
#     "id": 3,
#     "time": "2018-03-02T10:34:30.000Z"
#     "sourceAccount": "A",
#     "targetAccount": "B",
#     "amount": 100,
#     "category": "eating_out",
#   },


# [ [t1,t2,t3], [t4,t5,t6]]


# [t1,t2,t3] 

# proposta: seugnda etapa de processamento, que quebra um dos grupos gerados na etapa 1 em 1 ou mais grupos, pelo criterio do tempo
# para isso, para todo grupo:

# 1) Ordena o grupo por tempo
# 2) fixa o primeiro elemento do grupo, e vai avançando.
#  2.a) se a diferenca de tempo entre o el atual e o prox <= 1m, junta os caras
#   2. b) caso contrario, segrega num grupo separado e repete o algoritmo pro resto do grupo


# [1:00, 2:00, 4:00] => [1,2] , [4]
 

# [t1(1), t2(2), t3(4)]
#                  ^
# [t1,t2] 
# [t3]
# ^

# [t1]

def transacoes_juntas(t1, t2):
  return abs(t1.time - t2.time) <= 60s

def segrega_transacoes(lista_transactions):

  if len(lista_transactions) == 1:
    return lista_transactions

  pos_inicial = 0
  resposta = [lista_transactions[0]]

  while pos_inicial + 1 < len(lista_transactions):
    if (transacoes_juntas[lista_transactions[pos_inicial], lista_transactions[pos_inicial+1]]):
      resposta.append(lista_transactions[pos_inicial+1])
      pos_inicial+=1
    else:
      break

  resto_resposta = []
  if pos_inicial < len(lista_transactions):
    resto_resposta = segrega_transacoes(lista_transactions[pos_inicial:-1])

  return junta_listas(resposta, resto_resposta) # junta_lista faz o merge de duas lists



def agrupa_transacoes(transactions): # transactions = lista de transaction
  grupos = {}
  for transaction in transactions:
    id = transaction.id

    conteudo = {
      "amount": transaction.amount,
      "category": transaction.category,
      "sourceAccount": transaction.sourceAccount,
      "targetAccount": transaction.targetAccount
    }

    hash_conteudo = hash(conteudo) #funcao magica que me da hash

    if hash_conteudo not in grupos:
      grupos[hash_conteudo] = []

    grupos[hash_conteudo].append(transaction)

  resposta = []
  for grupo in grupos:
    resposta.append(grupo.value) #pega todas 

  transacoes_ordenadas = sort(lista_transactions) # finge que isso aqui retorna o cara ordenado por tempo crescente

  return segrega_transacoes(transacoes_ordenadas) # quebra os grupo por tempo
  


# hashmap <int, vector<int>> grupos =: chave = hash grupo, conteudo = os ids do grupo em si


# para cada transacao em transactions:
#     hash_transacao = hash(transaction)
#     id = transaction.id
#     grupos[hash_transacao].push_back(id)

# -> grupos vai ter todas as transactions agrupadas


# ideia inicial : 
# * tirar um hash dos campos sourceAcc, targetAcc, amount e cat => um valor que representa unicamente uma transacao
# * hash => o grupo que deve agrupar transacoes iguais



# /*
# Vamos fazer uma API/metodo que recebe uma lista de transacoes do gateway

# objetos nao necessariamente vem ordenados por id

# problema 1:

# Objeto:
#  {
#     "id": 3,
#     "time": "2018-03-02T10:34:30.000Z"
#     "sourceAccount": "A",
#     "targetAccount": "B",
#     "amount": 100,
#     "category": "eating_out",
#   },
#   algumas transactions podem estar duplicadas
#   - detectar transacoes duplicadas
#   - sorce acct, target, amount e category iguais
#   - retorno: vector<vector<transaction>>, com transactions iguais agrupadas
# */

# problema 2:
# segregue posteriormente a resposta do problema 1 assumindo que transactions com 1m de diferença sao iguais, e mais que isso sao diferentes