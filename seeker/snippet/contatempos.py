#date: 2022-02-07T16:58:32Z
#url: https://api.github.com/gists/b775e1b985124ce9e55af323f24079ec
#owner: https://api.github.com/users/Arduinobymyself

'''
Este arquivo cria o módulo ContaTempos e as listas usadas para efetuar testes
nos ordenadores do curso de Python da Coursera-2
deve ser salvo com o nome contatempos.py
será chamado no módulo de testes
'''

import random
import time
import ordenador # importa classe Ordenador() do arquivo ordenador.py

class ContaTempos:
  def lista_aleatoria(self, n):
    lista = [random.randrange(1000) for x in range(n)]
    return lista


  def lista_ordenada(self, n):
    lista = [x for x in range(n)]
    return lista


  def lista_quase_ordenada(self, n):
    lista =  [x for x in range(n)]
    lista[n//10] = -500
    return lista


  def compara(self, n):
    lista1 = self.lista_aleatoria(n)
    lista2 = lista1[:]
    lista3 = lista1[:]

    o = ordenador.Ordenador() # cria instância da classe Ordenador no arquivo ordenador.py

    # Listas aleatórias
    print('Comparando com listas aleatórias')
    antes = time.time()
    o.bolha(lista1)
    depois = time.time()
    print(f'Bolha demorou {depois - antes}')

    antes = time.time()
    o.bolha_melhorada(lista2)
    depois = time.time()
    print(f'Bolha melhorada demorou {depois - antes}')

    antes = time.time()
    o.selecao_direta(lista3)
    depois = time.time()
    print(f'Seleção Direta demorou {depois - antes}')

    print()

    lista1 = self.lista_quase_ordenada(n)
    lista2 = lista1[:]
    lista3 = lista1[:]

    # Listas quase ordenadas
    print('Comparando com listas quase ordenadas')
    antes = time.time()
    o.bolha(lista1)
    depois = time.time()
    print(f'Bolha demorou {depois - antes}')

    antes = time.time()
    o.bolha_melhorada(lista2)
    depois = time.time()
    print(f'Bolha melhorada demorou {depois - antes}')

    antes = time.time()
    o.selecao_direta(lista3)
    depois = time.time()
    print(f'Seleção Direta demorou {depois - antes}') 
