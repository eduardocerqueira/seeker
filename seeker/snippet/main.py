#date: 2022-12-20T16:58:35Z
#url: https://api.github.com/gists/d50a52485b2610a42bdb5314b32da782
#owner: https://api.github.com/users/JEdmario16

# vendas_via_web_reduzidissimo1.csv


def remove_aspas(valor):

    valor = valor.replace("'", "")  # remove aspas simples por nada
    valor = valor.replace('"', "")  # remove aspas duplas por

    return valor


def processa_linha(linha):
    linha_lista = linha.split(",")  # Separa todas as colunas da linha
    linha_tratada = []
    for col in linha_lista:
        linha_tratada.append(remove_aspas(col))

    # Transforma a coluna de preço para float
    # e de quantidade para int
    # Coluna 3 = quantidade
    # coluna 5 = preço unitário
    try:
        linha_tratada[3] = int(linha_tratada[3])
        linha_tratada[5] = float(linha_tratada[5])
    except ValueError:
        pass
    return linha_tratada


def leitura_arquivo(nome_arq):

    mat = []
    try:
        with open(nome_arq, "r", encoding="utf-8") as apont_arq:
            linhas = apont_arq.readlines()  # Vetor com linhas
            header = processa_linha(linhas[0])

            for linha in linhas[1:]:
                linha_tratada = processa_linha(linha)
                mat.append(linha_tratada)
        return (header, mat)

    except FileNotFoundError as exc:
        raise exc


def imprime_linha(linha):
    # Parto da premissa de que a ``linha`` passou pela função ``processa_linha``
    col1, col2, col3, col4, col5, col6, col7, col8 = linha
    print("%13s | %14s | %32s | %10s | %18s | %14s | %13s | %s" % (col1, col2, col3, col4, col5, col6, col7, col8)) 


def imprime_tabela(header, matriz_str):
    
    # Imprime o header
    imprime_linha(header)
    for linha in matriz_str:
        imprime_linha(linha)
