#date: 2022-04-06T16:59:19Z
#url: https://api.github.com/gists/d811ee568a361e7f858bda7f34952d16
#owner: https://api.github.com/users/jurandysoares

#!/bin/bash

## Problema
# Desafio para turma de ASA 2021.2, IFRN campus Parnamirim

# 1. Faça um script para criar uma conta para cada um dos nomes listados em [1] 
#    e que tenham o sobrenome1. Insira cada um dos usuários no grupo sobrenome1.
#
# 2. Faça um script para criar uma conta para cada um dos nomes listados em [1] 
#    e que tenham o sobrenome2. Insira cada um dos usuários no grupo sobrenome2.

# [1]: https://raw.githubusercontent.com/jurandysoares/eleitores-ifrn-2019/master/csv/alunos.csv

## Solução 1
# Criar usuários com nome de usuário (username) sequenciais

# Vetor com os sobrenomes desejados
SOBRENOMES=(veloso soares)
DOMINIO_EMAIL="redes.lab" # Trocar pelo domínio de seu
CSV_ALUNOS="alunos.csv"
URL_ALUNOSBD=https://bicharada.oulu.ifrn.edu.br/eleitores/alunos.txt
curl -so $CSV_ALUNOS $URL_ALUNOSBD

cria_usuario() {
    local usuario="$1"
    local nome_completo="$2"
    local email="$3"

    local prim_nome="${nome_completo//[ ]*/}"
    local sobrenome="${nome_completo#* }"

    if [ $# -eq 3 ]; then
        sudo samba-tool user create --given-name="${prim_nome}" --surname="${sobrenome}" --mail="${email}" "${usuario}"
    else
        echo "$0 requer 3 argumentos."
    fi

}

cria_grupo() {
    local grupo="$1"

    if [ $# -eq 1 ]; then
        sudo samba-tool group create "$grupo"
    else
        echo "$0 requer 1 argumentos."
    fi
}

insere_usuario_em_grupos() {
    local usuario="$1"
    IFS=, read -ra grupos <<<"$2"

    if [ $# -eq 2 ]; then
        for g in "${grupos[@]}"; do
            sudo samba-tool group addmembers "${g}" "${usuario}"
        done
    else
        echo "$0 requer 2 argumentos."
    fi

}


# Descarta a 1ª linha, com título das colunas: tail +2
# Extrai o 3ª campo: Opção -f3 do cut
# Considerando "," como delimitador: Opção -d, do cut
# Ordena deixando somente os elementos únicos: sort -u
campi=$(tail +2 $CSV_ALUNOS | cut -d, -f3 | sort -u)

# Dicionário ou arranjo associativo (*associative array*, em inglês)
declare -A contador_campus

# Cria um grupo para cada campus
# e inicia contador de usuário do campus em 1
for campus in $campi; do
    cria_grupo "G_${campus}" 
    contador_campus[$campus]=1
done

## Criar um grupo para cada sobrenome
for sobrenome in "${SOBRENOMES[@]}"; do
    cria_grupo "G_${sobrenome}"
done

# Usuários com sobrenome "veloso" == ${SOBRENOMES[0]}
for sobrenome in "${SOBRENOMES[@]}"; do

    while IFS=, read -r _id nome campus; do
        username="usuario-${campus,,}-${contador_campus[$campus]}"
        cria_usuario "$username" "${nome}" "${username}@${DOMINIO_EMAIL}"
        #            \_________/ \_______/ \____________________________/
        #              Usuário      Nome         E-mail
        #
        # https://github.com/koalaman/shellcheck/wiki/SC2219
        insere_usuario_em_grupos "${username}" "G_${campus},G_${sobrenome}"
        (( contador_campus[$campus]++ )) || true
    done < <(grep -wi "${sobrenome}" $CSV_ALUNOS)
done