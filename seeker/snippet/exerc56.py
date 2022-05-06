#date: 2022-05-06T17:08:43Z
#url: https://api.github.com/gists/fc63a7e1f79e89f887ca7b119d7d16ee
#owner: https://api.github.com/users/sklarow

def main():

    somaIdade = 0
    nomeHomemMaisVelho = ""
    idadeHomemMaisVelho = 0
    qtdMulheresMenosDe20Anos = 0

    for i in range(4):
        print(f"Pessoa #{i+1}")
        nome = str(input("Nome: "))
        idade = int(input("Idade: "))
        sexo = str(input("Sexo [M/F]:"))

        somaIdade = somaIdade + idade

        if sexo == "F" and idade < 20:
            qtdMulheresMenosDe20Anos = qtdMulheresMenosDe20Anos + 1
        
        if sexo == "M" and idade > idadeHomemMaisVelho:
            nomeHomemMaisVelho = nome
            idadeHomemMaisVelho = idade
    
    mediaIdade = somaIdade/4
    print(f"Média de idade: {mediaIdade}")
    if nomeHomemMaisVelho:
        print(f"O nome do homem mais velho é: {nomeHomemMaisVelho} e ele tem {idadeHomemMaisVelho} anos de idade")
    print(f"Temos um total de {qtdMulheresMenosDe20Anos} mulher(es) com menos de 20 anos")

main()