#date: 2023-10-06T17:00:09Z
#url: https://api.github.com/gists/f60f19432263f87d8f7130dac9fb5af3
#owner: https://api.github.com/users/maxwellamaral

import os
import sys
from datetime import datetime
import shutil

PASTA_PADRAO = "./00-09 Meta/00 Indice/00.02 Estrutura"


def listar_pastas(diretorio, profundidade=0, nivel_max=3):  
    # Função para listar pastas e subpastas até um nível máximo
    # Retorna uma lista com os caminhos completos das pastas encontradas
    pastas = []  # Inicializa a lista de pastas

    if profundidade > nivel_max:
        return pastas

    try:
        lista = os.listdir(diretorio)
    except PermissionError:
        # Ignorar erros de permissão
        return pastas

    subpastas = [
        os.path.join(diretorio, pasta)
        for pasta in lista
        if os.path.isdir(os.path.join(diretorio, pasta)) and pasta[0].isdigit()
    ]

    for pasta in subpastas:
        pastas.append(pasta)  # Adiciona a pasta à lista de pastas
        pastas.extend(listar_pastas(pasta, profundidade + 1, nivel_max))

    return pastas  # Retorna a lista de pastas


def criar_pasta_estrutura(caminho_pai, nome_pasta):
    # Cria uma pasta no caminho especificado
    caminho_estrutura = os.path.join(caminho_pai, nome_pasta)
    os.makedirs(
        caminho_estrutura, exist_ok=True
    )  # Cria a pasta ou ignora se já existir


def criar_arquivo_markdown(diretorio_inicial, nome_arquivo, pastas):
    # Cria um arquivo Markdown com a estrutura de árvore em formato de bloco de código
    with open(nome_arquivo, "w", encoding="utf-8") as arquivo_md:
        data_atual = datetime.now().strftime("%d/%m/%Y")
        arquivo_md.write(f"Criado em {data_atual}\n\n")  # Inicia o bloco de código

        arquivo_md.write("```\n")  # Inicia o bloco de código
        for pasta in pastas:
            nivel = pasta.count(os.sep) - diretorio_inicial.count(os.sep)
            linha = "|   " * nivel + "|--- " + os.path.basename(pasta) + "\n"
            arquivo_md.write(linha)
        arquivo_md.write("```\n")  # Finaliza o bloco de código


if __name__ == "__main__":
    # Recebe o caminho do diretório inicial como argumento de linha de comando
    if len(sys.argv) < 2:
        print("Por favor, forneça o caminho do diretório como argumento.")
    else:
        # Obtem o caminho do diretório inicial do argumento de linha de comando
        diretorio_inicial = sys.argv[1]
        data_atual = datetime.now().strftime("%Y%m%d")
        nome_pasta_estrutura = f"{data_atual} estrutura"
        nome_arquivo_markdown = os.path.join(PASTA_PADRAO, f"{data_atual} estrutura.md")
        nome_arquivo_zip = os.path.join(PASTA_PADRAO, f"{data_atual} estrutura.zip")

        # Obter a lista de pastas
        pastas = listar_pastas(diretorio_inicial)

        # Criar o arquivo Markdown com a estrutura de árvore em formato de bloco de código
        criar_arquivo_markdown(diretorio_inicial, nome_arquivo_markdown, pastas)
        print(f"Arquivo Markdown '{nome_arquivo_markdown}' criado com sucesso.")

        # Criar a pasta 'yyyymmdd estrutura'
        pasta_destino = input(
            f"Digite o caminho da pasta de destino (ou pressione Enter para usar o padrão '{PASTA_PADRAO}'): "
        )
        if not pasta_destino:
            pasta_destino = PASTA_PADRAO

        pasta_estrutura = os.path.join(pasta_destino, nome_pasta_estrutura)
        criar_pasta_estrutura(pasta_destino, nome_pasta_estrutura)

        # Criar as pastas listadas pela função listar_pastas dentro da nova pasta
        for pasta in pastas:
            caminho_relativo = os.path.relpath(pasta, diretorio_inicial)
            caminho_pasta_nova = os.path.join(pasta_estrutura, caminho_relativo)
            os.makedirs(caminho_pasta_nova, exist_ok=True)

        # Mover o conteúdo da pasta 'yyyymmdd estrutura' para um arquivo compactado zip
        shutil.make_archive(
            nome_arquivo_zip[:-4], "zip", pasta_destino, nome_pasta_estrutura
        )

        # Excluir a pasta 'yyyymmdd estrutura'
        shutil.rmtree(pasta_estrutura)

        print(
            f"Conteúdo da pasta '{nome_pasta_estrutura}' movido para '{nome_arquivo_zip}' e a pasta foi excluída com sucesso."
        )
