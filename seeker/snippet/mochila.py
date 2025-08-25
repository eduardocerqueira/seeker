#date: 2025-08-25T17:06:17Z
#url: https://api.github.com/gists/83de0dc55336fbe1cd0d324ccc07d046
#owner: https://api.github.com/users/bielelias

from mip import *
import random

# Constantes
RANDOM_SEED   = 1
NUM_PRODUTOS  = 10
NUM_MOCHILAS  = 2

# Variável que armazena os Dados
produtos = {}
mochilas = {}
gerar_produtos(produtos)
gerar_mochilas(mochilas)

# Todos os produtos
imprimir_produtos(produtos)

# Todas as mochilas
imprimir_mochilas(mochilas)

# Modelo 
modelo = Model(sense=MAXIMIZE)

# Variáveis de decisão
carga = {} # qual produto será colcoado em cada mochila

for m in mochilas:
    for p in produtos:
        carga[(m, p)] = modelo.add_var(var_type=BINARY)     
        
# Restrições
# a) O mesmo produto não pode ser colocado nas duas mochilas
for p in produtos:        
    modelo += xsum(carga[(m, p)] for m in mochilas) <= 1

# b) A soma dos pesos dos produtos alocados em uma mochila não devem ser maior do que a carga máxima suportada pela mochila
for m in mochilas:    
    modelo += xsum(carga[(m, p)] * produtos[p]['peso'] for p in produtos) <= mochilas[m]['carga_maxima']

# Função Objetivo
modelo.objective = maximize(
    xsum(carga[(m, p)] * produtos[p]['valor']
         for m in mochilas 
             for p in produtos
         if (m, p) in carga
    )
)

modelo.optimize()

# Resultado
print("\n=====")
print("Valor Total em Todas as Mochilas {}".format(modelo.objective_values))
print("=====")
for m in mochilas:
    print("\nCarga da Mochila {} com capacidade de {}g".format(m, mochilas[m]['carga_maxima']))
    valor_total = 0
    carga_total = 0
    for p in produtos:
        if (carga[(m, p)].x == 1):
            valor_total += produtos[p]['valor']
            carga_total += produtos[p]['peso']
            print("{} \tR$ {},00 \t{}g".format(p, produtos[p]['valor'], produtos[p]['peso']))
    
    print("-\nValor Total: R$ {},00 \nCarga Total: {}g\nCapacidade Ociosa: {}g".format(valor_total, carga_total, (mochilas[m]['carga_maxima']-carga_total)))

# Gerar produtos com peso e valor aleatórios
def gerar_produtos(prod):
    random.seed(RANDOM_SEED)
    for i in range(NUM_PRODUTOS):
        cod = 'p_{}'.format(i)
        valor = random.choice(range(1,10)) # Valor em reais
        peso = random.choice(range(100,999)) # Valor em reais
        prod[cod] = {
            'valor': valor,
            'peso': peso
        }
# Imprimir os produtos
def imprimir_produtos(prod):
    print("PRODUTOS")
    print("Cód \tValor \t\tPeso")
    for p in prod:
        print("{}\tR$ {},00 \t{}g".format(p, prod[p]['valor'], prod[p]['peso']))
        
# Gera mochilas com pesos aleatórias
def gerar_mochilas(moc):
    random.seed(RANDOM_SEED)
    for i in range(NUM_MOCHILAS):
        cod = 'm_{}'.format(i)
        carga_maxima = random.choice(range(500,2000)) # Carga máxima em gramas
        moc[cod] = {
            'carga_maxima': carga_maxima
        }

# Imprime as mochilas
def imprimir_mochilas(moc):
    print("\nMOCHILAS")
    print("Cód \tCarga Máxima")
    for m in moc:
        print("{} \t{}g".format(m, moc[m]['carga_maxima']))