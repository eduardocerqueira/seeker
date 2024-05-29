#date: 2024-05-29T17:09:22Z
#url: https://api.github.com/gists/6e825255bd78c7f7f50f29b0ab00067a
#owner: https://api.github.com/users/Tg7vg

while True:
    try:
        preco = float(input("Preço do Produto: R$"))
        if preco < 0.0000000001:
            print("Insira um valor funcional.")
        else:
            break
    except ValueError:
        print("Insira somente o preço do produto")
while True:
    try:
        decisao = input("Defina parcelado ou á vista: ")
        if decisao in ["PARCELADO","parcelado","Parcelado","PARCEL","Parcel","parcel","P","p"] or ["V","v","á vista","Á VISTA","Á vista","Á Vista","vista","Vista","VISTA"]:
            break
        else:
            print("Digite somente parcelado ou á vista.")
    except ValueError:
        print("Somente parcelado ou á vista: ")

if decisao in ["V","v","á vista","Á VISTA","Á vista","Á Vista","vista","Vista","VISTA"]:
    while True:
        try:
            valor_do_desconto = float(input("Quanto de desconto você quer aplicar? "))
            if valor_do_desconto < 0:
                print("Não é possível aplicar desconto com este valor.")
            elif valor_do_desconto > 100:
                print("Não é possível aplicar desconto com este valor.")
            else:
                break
        except ValueError:
            print("Coloque apenas valores de 0%""a 100%")
    
    desconto = valor_do_desconto / 100
    descontoaplicado = preco * desconto
    novopreco = preco - descontoaplicado
    print(f"Seu novo preço com desconto será: R${novopreco:.2f}")

elif decisao in ["PARCELADO","parcelado","Parcelado","PARCEL","Parcel","parcel","P","p"]:

    while True:
        try:
            escolha = input("Juros Simples ou Compostos? ")

            if escolha in ["SIMPLES","Simples","simples","S","s","Juros Simples","JUROS SIMPLES","juros simples","js","JS","Juros simples","Js","S","s"]:

                parcel = int(input("Parcelar em quantas vezes: "))
                while True:
                    try:
                        if parcel < 1:
                            print("Digite somente em quantas vezes será parcelado.")
                        else:
                            break
                    except ValueError:
                        print("Digite somente valores numéricos")
                
                while True:
                    try:
                        taxaS = float(input("Qual a taxa de juros: "))
                        if taxaS < 0:
                            print("Somente valores positivos e inteiros. ")
                        else:
                            break
                    except ValueError:
                        print("Digite somente valores numéricos")
                
                juros =  preco * (taxaS/100) * parcel
                montante = preco + juros
                

                print(f"Ao mês: R${juros:.2f}")
                print(f"Total com juros: R${montante:.2f}")
                break

            elif escolha in ["Compostos","COMPOSTOS","compostos","JC","Jc","jc","COMPOSTO","Composto","composto","JUROS COMPOSTOS","JUROS COMPOSTO","Juros Compostos","Juros Composto","Juros compostos","Juros composto","juros compostos","juros composto","C","c"]:
                parcel = int(input("Parcelar em quantas vezes: "))
                while True:
                    try:
                        if parcel < 1:
                            print("Digite somente em quantas vezes será parcelado.")
                        else:
                            break
                    except ValueError:
                        print("Digite somente valores numéricos")

                while True:
                    try:
                        taxaC = float(input("Qual a taxa de juros: "))
                        if taxaC < 0:
                            print("Somente valores positivos e inteiros. ")
                        else:
                            break
                    except ValueError:
                        print("Digite somente valores numéricos")

                pormês = (preco) / ((((1 + (taxaC/100) )**parcel)-1) / (((1 + (taxaC/100))**parcel) * (taxaC/100)))
                montante = pormês * parcel

                print(f"Ao mês: R${pormês:.2f}")
                print(f"Total com juros: R${montante:.2f}")
                break

            else:
                print("Escolha Juros Simples ou Compostos")
        except ValueError:
            print("Somente Simples ou Composto")

else:
    print("Escolha somente parcelado ou á vista")


input("Aperte enter para sair...")