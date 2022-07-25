#date: 2022-07-25T17:08:03Z
#url: https://api.github.com/gists/eae92d906c3064735bb03421cf9ffc4b
#owner: https://api.github.com/users/elderlourenzi

# algoritmo para conferência de gabarito. Sem tratamento de "type errors".

print("Bem-vindo")
#Recebe o número de questões que compõe a prova
qtQuest = int(input("Quantas questões compõe sua avaliação? "))

#Recebe o gabarito da prova e armazena em lista
gabaProf = list(str.upper(input('Digite o gabarito da prova: ')))

#Conta o número de respostas digitadas no gabarito
qtgabaProf = len(gabaProf)

#testa se o número de respostas digitadas é o mesmo que a quantidade de questões da prova.
while qtgabaProf != qtQuest:
    print("A quantidade de repostas não corresponde ao número de questões da prova")
    gabaProf = list(str.upper(input('Digite o gabarito da prova: ')))
    qtgabaProf = len(gabaProf)

#Recebe o gabarito do aluno e armazena em lista
gabaAluno = list(str.upper(input("Digite as respostas do aluno: ")))

#Conta o número de respostas digitadas
qtQuestAl = len(gabaAluno)

#Testa se o número de repostas do gabarito do aluno é o mesmo do gabarito do professor
while qtQuestAl != qtQuest:
    print("O número de respostas digitadas para o gabarito do aluno não correponde ao número de questões da prova")
    gabaAluno = list(str.upper(input("Digite as respostas do aluno: ")))
    qtQuestAl = len(gabaAluno)

#Percorre os gabaritos procurando por combinações, a cada acerto soma +1 na váriavél "acerto".
acertos = 0
for i in range(len(gabaProf)):
    if gabaAluno[i] == gabaProf[i]:
        acertos += 1

#calcula o peso de cada questão baseado na quantidade de questões da prova
pesoQ = (float(10/qtQuest))
#gera a nota da prova:
notaProva = (pesoQ*acertos)

#imprime a quantidade de acertos, a nota do aluno e os dois gabaritos.
print(f'O aluno obteve {acertos} acertos, sua nota é {notaProva}')
print(f' Gabarito da prova: {gabaProf} \n Gabarito do aluno: {gabaAluno}')