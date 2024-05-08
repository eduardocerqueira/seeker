#date: 2024-05-08T16:42:25Z
#url: https://api.github.com/gists/a50572bccbffee6e23a959c6cbec6d29
#owner: https://api.github.com/users/CoimbraVitor

import struct

def cria_arq():
    with open('vendedores.bin', 'wb'):
        pass

def incluir_vendedor(bf, codigo_vendedor, valor_venda, mes_ano, deleted=False):
    bf.seek(0)
    while True:
        codigo_atual = struct.unpack('i', bf.read(4))[0]
        if codigo_atual == 0:
            break

    bf.write(struct.pack('i', codigo_vendedor))
    bf.write(struct.pack('f', valor_venda))
    bf.write(bytes(mes_ano, 'latin-1'))
    bf.write(struct.pack('?', deleted))

def excluir_vendedor(bf, codigo_vendedor):
    bf.seek(0)
    while True:
        try:
            codigo_atual = struct.unpack('i', bf.read(4))[0]
            if codigo_atual == codigo_vendedor:
                bf.seek(bf.tell() + 7)
                bf.write(struct.pack('?', True))
            else:
                bf.seek(bf.tell() + 8)
        except struct.error:
            break

def alterar_valor_venda(bf, codigo_vendedor, mes_ano, valor_alterado):
    bf.seek(0)
    while True:
        try:
            codigo_atual = struct.unpack('i', bf.read(4))[0]
            mes_ano_atual = bf.read(7).decode('latin-1')
            deleted = struct.unpack('?', bf.read(1))[0]
            if codigo_atual == codigo_vendedor and mes_ano_atual == mes_ano and not deleted:
                bf.seek(bf.tell() - 12)
                bf.write(struct.pack('f', valor_alterado))
                break
        except struct.error:
            break

def imprimir_registros(bf):
    bf.seek(0)
    while True:
        try:
            codigo_vendedor = struct.unpack('i', bf.read(4))[0]
            valor_venda = struct.unpack('f', bf.read(4))[0]
            mes_ano = bf.read(7).decode('latin-1')
            deleted = struct.unpack('?', bf.read(1))[0]
            if not deleted:
                print(codigo_vendedor, valor_venda, mes_ano)
        except struct.error:
            break

def consultar_maior_valor(bf):
    bf.seek(0)
    maior_valor = 0
    while True:
        try:
            codigo_vendedor = struct.unpack('i', bf.read(4))[0]
            valor_venda = struct.unpack('f', bf.read(4))[0]
            mes_ano = bf.read(7).decode('latin-1')
            deleted = struct.unpack('?', bf.read(1))[0]
            if not deleted and valor_venda > maior_valor:
                maior_valor = valor_venda
                posicao_do_maior = bf.tell()
        except struct.error:
            break
    bf.seek(posicao_do_maior - 16)
    codigo_vendedor = struct.unpack('i', bf.read(4))[0]
    valor_venda = struct.unpack('f', bf.read(4))[0]
    mes_ano = bf.read(7).decode('latin-1')

    print(f"O Vendedor com maior valor da venda é {codigo_vendedor} {valor_venda} {mes_ano}")

def main():
    while True:
        print("1 - Criar o arquivo de dados\n2 - Incluir um determinado vendedor no arquivo\n3 - Excluir um determinado vendedor no arquivo\n4 - Alterar o valor total da venda de um determinado vendedor de um determinado mês\n5 - Imprimir os registros na saída padrão\n6 - Consultar o vendedor com maior valor da venda\n7 - Finalizar o programa")
        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            cria_arq()
            print("Novo arquivo de dados criado!")
        elif opcao == '2':
            codigo_vendedor = int(input("Código do vendedor: "))
            valor_venda = float(input("Valor da venda: "))
            mes_ano = input("Data da venda: (mm/aaaa) ")
            with open("vendedores.bin", "ab+") as bf:
                incluir_vendedor(bf, codigo_vendedor, valor_venda, mes_ano)
        elif opcao == '3':
            codigo_vendedor = int(input("Qual o código do vendedor que deseja deletar: "))
            with open("vendedores.bin", "rb+") as bf:
                excluir_vendedor(bf, codigo_vendedor)
        elif opcao == '4':
            codigo_vendedor = int(input("Qual o código do vendedor que deseja alterar: "))
            mes_ano = input("Qual data da venda (mm/aaaa): ")
            valor_alterado = float(input("Qual o novo valor: "))
            with open("vendedores.bin", "rb+") as bf:
                alterar_valor_venda(bf, codigo_vendedor, mes_ano, valor_alterado)
        elif opcao == '5':
            with open("vendedores.bin", "rb") as bf:
                imprimir_registros(bf)
        elif opcao == '6':
            with open("vendedores.bin", "rb+") as bf:
                consultar_maior_valor(bf)
        elif opcao == '7':
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
