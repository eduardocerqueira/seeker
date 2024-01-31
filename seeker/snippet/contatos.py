#date: 2024-01-31T16:55:58Z
#url: https://api.github.com/gists/ac5bc1b8d9c2d8c9b8cfe1724dcf393b
#owner: https://api.github.com/users/victineo

def adicionar_contato(lista_contatos):
    print("\nO contato deve possuir nome, número de telefone e e-mail.")
    nome_contato = input("Insira o nome do contato: ")
    telefone_contato = input("Insira o número de telefone do contato: ")
    email_contato = input("Insira o email do contato: ")
    contato = {"nome": nome_contato, "telefone": telefone_contato, "e-mail": email_contato, "favoritado": False}
    lista_contatos.append(contato)
    print(f"\n'{nome_contato}' adicionado aos contatos.")

def ver_contatos(lista_contatos):
    while True:
        print("\nLista de contatos:")
        if not lista_contatos:
            print("Ainda não há nenhum contato para exibir.")
            break
        else:
            for indice, contato in enumerate(lista_contatos, start=1):
                status = "★" if contato["favoritado"] else "☆"
                nome_contato = contato["nome"]
                print(f"{indice}. {nome_contato} {status}")

            escolha_ver = input("\nDigite o índice de um contato para ver suas informações, ou 'voltar' para voltar ao menu principal: ")

            if escolha_ver.isdigit():
                indice_contato_ajustado = int(escolha_ver) - 1
                if 0 <= indice_contato_ajustado < len(lista_contatos):
                    contato_selecionado = lista_contatos[indice_contato_ajustado]
                    print(f"\nNome: {contato_selecionado['nome']}\nTelefone: {contato_selecionado['telefone']}\nE-mail: {contato_selecionado['e-mail']}")
                else:
                    print("\nÍndice de contato inválido.")
            elif escolha_ver.lower() == "voltar":
                break
            else:
                print("\nComando inválido.")

def editar_contato(lista_contatos):
    print("\nLista de contatos:")
    if not lista_contatos:
        print("Ainda não há nenhum contato para editar.")
    else:
        for indice, contato in enumerate(lista_contatos, start=1):
            status_icone = "★" if contato["favoritado"] else "☆"
            nome_contato = contato["nome"]
            print(f"{indice}. {nome_contato} {status_icone}")
        indice_contato = input("\nDigite o índice do contato que deseja editar: ")
        indice_contato_ajustado = int(indice_contato) - 1

        if indice_contato_ajustado < 0 or indice_contato_ajustado >= len(lista_contatos):
            print("Índice de contato inválido.")
            return
        
        contato_selecionado = lista_contatos[indice_contato_ajustado]

        while True:
            print("\nOpções:")
            print("1. Editar nome do contato")
            print("2. Editar número de telefone do contato")
            print("3. Editar e-mail do contato")
            print("4. Concluir")
            escolha_editar = input("Insira sua escolha: ")

            if escolha_editar == "1":
                novo_nome_contato = input("\nInsira um novo nome para o contato: ")
                contato_selecionado["nome"] = novo_nome_contato
                print(f"\nNome do contato atualizado para '{novo_nome_contato}'.")
            elif escolha_editar == "2":
                novo_telefone_contato = input("\nInsira um novo número de telefone para o contato: ")
                contato_selecionado["telefone"] = novo_telefone_contato
                print(f"\nNúmero de telefone atualizado para '{novo_telefone_contato}'.")
            elif escolha_editar == "3":
                novo_email_contato = input("\nInsira um novo e-mail para o contato: ")
                contato_selecionado["e-mail"] = novo_email_contato
                print(f"\nE-mail atualizado para '{novo_email_contato}'.")
            elif escolha_editar == "4":
                break

def favoritar_contato(lista_contatos):
    print("\nLista de contatos:")
    if not lista_contatos:
        print("Ainda não há nenhum contato para favoritar ou desfavoritar.")
    else:
        for indice, contato in enumerate(lista_contatos, start=1):
            status_icone = "★" if contato["favoritado"] else "☆"
            nome_contato = contato["nome"]
            print(f"{indice}. {nome_contato} {status_icone}")

        indice_contato = input("\nDigite o índice do contato a ser favoritado/desfavoritado: ")
        indice_contato_ajustado = int(indice_contato) - 1

        if 0 <= indice_contato_ajustado < len(lista_contatos):
            contato_selecionado = lista_contatos[indice_contato_ajustado]

            contato_selecionado["favoritado"] = not contato_selecionado["favoritado"]
            status_favorito = "favoritado" if contato_selecionado["favoritado"] else "desfavoritado"
            print(f"\nO contato '{contato_selecionado['nome']}' foi {status_favorito}.")
        else:
            print("Índice de contato inválido.")

def ver_contatos_favoritos(lista_contatos):
    while True:
        print("\nLista de contatos favoritados:")
        contatos_favoritados = [contato for contato in lista_contatos if contato['favoritado']]
        if not contatos_favoritados:
            print("Ainda não há nenhum contato favoritado para exibir.")
            break
        else:
            for indice, contato in enumerate(contatos_favoritados, start=1):
                status_icone = "★" if contato["favoritado"] else "☆"
                nome_contato = contato["nome"]
                print(f"{indice}. {nome_contato} {status_icone}")
            
            escolha_ver_favorito = input("\nDigite o índice de um contato favoritado para ver suas informações, ou 'voltar' para voltar ao menu principal: ")

            if escolha_ver_favorito.isdigit():
                indice_favorito_ajustado = int(escolha_ver_favorito) - 1
                if 0 <= indice_favorito_ajustado < len(contatos_favoritados):
                    contato_selecionado = contatos_favoritados[indice_favorito_ajustado]
                    print(f"\nNome: {contato_selecionado['nome']}\nTelefone: {contato_selecionado['telefone']}\nE-mail: {contato_selecionado['e-mail']}")
                else:
                    print("Índice de contato inválido.")
            elif escolha_ver_favorito.lower() == "voltar":
                break

def apagar_contato(lista_contatos):
    print("\nLista de contatos:")
    if not lista_contatos:
        print("Ainda não há nenhum contato para apagar.")
    else:
        for indice, contato in enumerate(lista_contatos, start=1):
            status_icone = "★" if contato["favoritado"] else "☆"
            print(f"{indice}. {contato['nome']} {status_icone}")
        
        indice_contato = input("\nDigite o índice do contato que deseja apagar: ")
        if indice_contato.isdigit():
            indice_contato_ajustado = int(indice_contato) - 1
            confirmacao_excluir = input(f"Tem certeza de que deseja apagar '{lista_contatos[indice_contato_ajustado]['nome']}' dos seus contatos? (S/N): ")
            if confirmacao_excluir.upper() == "S":
                contato_removido = lista_contatos.pop(indice_contato_ajustado)
                print(f"\nO contato '{contato_removido['nome']}' foi apagado com sucesso.")
            elif confirmacao_excluir.upper() == "N":
                print("\nOperação cancelada.")
                return
            
            if 0 <= indice_contato_ajustado < len(lista_contatos):
                contato_removido = lista_contatos.pop(indice_contato_ajustado)
                print(f"\nO contato '{contato_removido['nome']}' foi apagado com sucesso.")
            else:
                print("Índice de contato inválido.")
        else:
            print("Índice de contato inválido.")

lista_contatos = []

while True:
    print("\nMenu do Gerenciador de contatos")
    print("1. Adicionar contato")
    print("2. Ver lista de contatos")
    print("3. Editar contato")
    print("4. Favoritar/Desfavoritar contato")
    print("5. Ver contatos favoritos")
    print("6. Apagar contato")
    escolha = input("\nDigite sua escolha: ")
    if escolha == "1":
        adicionar_contato(lista_contatos)
    elif escolha == "2":
        ver_contatos(lista_contatos)
    elif escolha == "3":
        editar_contato(lista_contatos)
    elif escolha == "4":
        favoritar_contato(lista_contatos)
    elif escolha == "5":
        ver_contatos_favoritos(lista_contatos)
    elif escolha == "6":
        apagar_contato(lista_contatos)