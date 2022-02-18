#date: 2022-02-18T16:57:39Z
#url: https://api.github.com/gists/6d9902339e624603478260d1f5362708
#owner: https://api.github.com/users/gabriellopesdesouza2002

import re
import random


REGRESSIVOS = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]# constante = variável que nunca vai mudar


def valida(cnpj):
    cnpj = apenas_numeros(cnpj)

    try:
        if eh_sequencia(cnpj):
            return False
    except:
        return False

    try:
        novo_cnpj = calcula_digito(cnpj=cnpj, digito=1)
        novo_cnpj = calcula_digito(cnpj=novo_cnpj, digito=2)
    except Exception as e:
        return False

    if novo_cnpj == cnpj:
        return True
    else:
        return False


def calcula_digito(cnpj, digito):
    if digito == 1: # se o digito for igual a 1
        regressivos = REGRESSIVOS[1:]  # começa a contar a partir do 5 = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2) e nao do 6
        novo_cnpj = cnpj[:-2]
    elif digito == 2:
        regressivos = REGRESSIVOS  # começa a partir do inicio da lista
        novo_cnpj = cnpj
    else: # se nao for nem 1 nem 2 retorna none 
        return None
    
    total = 0 # vai pegar a multiplicacao e dela pegar a soma
    for indice, regressivo in enumerate(regressivos): # vai pegar o indice de cada item e o valor de cada número do regressivo
        # acessa cada indice do cnpj
        total += int(cnpj[indice]) * regressivo  # vai pegar a multiplicacao e dela pegar a soma
    
    digito = 11 - (total % 11)
    digito = digito if digito <= 9 else 0
    # se o digito for menor ou igual a 9 recebe o valor do calculo, senao, recebe o valor 0
    return f'{novo_cnpj}{digito}'
    
def apenas_numeros(cnpj):
    '''
    ### Pega os números do cnpj e retorna... 
    #### ( remove todos os caracteres que forem diferentes de 0 a 9)
    '''
    return re.sub(r'[^0-9]', '', cnpj) # diferente de 0 a 9, subtitui por nada (deixando os números coladinhos)

# verifica se não é uma sequencia (111.111.111.111.111)
def eh_sequencia(cnpj):
    sequencia = cnpj[0] * len(cnpj) # cria uma sequencia, ex primeiro num = 0, (000000000000000)
    
    if sequencia == cnpj: # se for uma sequencia de números iguais (11111111111111)
        return True  # retorna true e fala que é invalido
    else:
        return False  # retorna false e fala que é VÁLIDO
    print(sequencia)


#### gerador ####
def gera():
    primeiro_digit = random.randint(0, 9)
    segundo_digit = random.randint(0, 9)
    segundo_bloco = random.randint(100, 999)
    terceiro_bloco = random.randint(100, 999)
    quarto_bloco = '0001'
    
    inicio_cnpj = f'{primeiro_digit}{segundo_digit}{segundo_bloco}{terceiro_bloco}{quarto_bloco}00'
    
    novo_cnpj = calcula_digito(cnpj=inicio_cnpj, digito=1)
    novo_cnpj = calcula_digito(cnpj=novo_cnpj, digito=2)
    
    return novo_cnpj

def formata(cnpj):
    cnpj = apenas_numeros(cnpj)
    formatado = f'{cnpj[0:2]}.{cnpj[2:5]}.{cnpj[5:8]}/{cnpj[8:12]}-{cnpj[12:]}'
    return formatado
  
  
  
  
  ####### Execute no main.py #############
  
  import cnpj
import os
os.system('cls' if os.name == 'nt' else 'clear')

'''
O módulo se chama cnpj, se eu chamar algo aqui dentro de cnpj, vai subistituir o módulo
'''
cnpj1 = "40.688.134/0001-61"


if cnpj.valida(cnpj1):  # se a funçao retornar True:
    print(f'{cnpj1} É válido')
else:
    print(f'{cnpj1} É inválido')


for i in range(100):  # cria 100 cnpj's
    novo_cnpj = cnpj.gera()
    formatado = cnpj.formata(novo_cnpj)
    print(formatado)
