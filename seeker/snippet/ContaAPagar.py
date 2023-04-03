#date: 2023-04-03T16:44:53Z
#url: https://api.github.com/gists/8917ac26dfac97825074bbdf0a49dde1
#owner: https://api.github.com/users/jaymestech

#definindo os preços de cada codigo de produto para ajudar na proxima etapa
precos= {1: 10.5, 2: 12, 3: 14, 4: 8, 5: 5.5}
#solicita qual o codigo do item atribuindo esse valor a variavel "codigo"
codigo = int(input("Informe o código do item: "))
#para o usuario não precisar digitar a quantidade de produtos de um um codigo que já não existe, ja é emitido a mensagem de erro antes4

if codigo not in precos:
    print("Código inválido")
    exit()
#solicita qual a quantidade de produtos
quantidade = int(input("Informe a quantidade do item: "))
#verifica qual valor é atribuido de acordo com o codigo informado
if codigo == 1:
    valor = 10.5
elif codigo == 2:
    valor = 12
elif codigo == 3:
    valor = 14
elif codigo == 4:
    valor = 8
elif codigo == 5:
    valor = 5.5
#se o codigo informado não constar no banco de dados aparecerá uma mensagem de erro e programa é encerrado
else:
    print("Código Inválido!")
    exit()
#calcula o total multiplicando o valor do item pela quantidade informada
total = valor * quantidade
print("Total = R$", total)