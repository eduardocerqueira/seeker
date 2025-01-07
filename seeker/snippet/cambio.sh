#date: 2025-01-07T16:59:01Z
#url: https://api.github.com/gists/737bcf8f1aedf11a46f0780b8515e627
#owner: https://api.github.com/users/reginaldosnunes

#!/bin/bash
# Nome do Script: cambio.sh
# Descrição: Este script exibe o valor atual do dolar e converte de USD para BRL
# com base na cotação fornecida pelo Banco Central do Brasil
# https://www.bcb.gov.br/estabilidadefinanceira/cotacoestodas
#
# Autor: Reginaldo Nunes <reginaldonunes@outlook.com>
# Versão: 1.0
# Data: 2025-01-06
# Licença: MIT
# Repositório: https://gist.github.com/reginaldosnunes
#
# Changelog
# [1.0.0] - 2025-01-06
# - Versão inicial do script.
# - Funcionalidades:
#   - Baixa o arquivo CSV do Banco Central.
#   - Converte USD para BRL com base na cotação do dia.
#
# Uso: ./cambio.sh ou ./cambio.sh 8.73
#

VERSION="1.0.0"
now=$(date +%Y%m%d)

get_calc() {
  # Verifica se $1 é um número válido, caso contrário, define o valor padrão como 1
  if [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    usd="$1"
  else
    usd=1
  fi

  # Extrai o valor do BRL do arquivo CSV
  brl=$(grep USD "/tmp/${now}.csv" | cut -d ";" -f 5 | tr ',' '.')

  # Calcula e exibe o resultado
  awk -v brl="$brl" -v usd="$usd" 'BEGIN {
    if (usd == 1) {
      printf "dolar hoje: R$ %.2f\n", brl
    } else {
      printf "valor convertido: USD %.2f = R$ %.2f\n", usd, brl * usd
    }
  }'
}

# Verifica se o arquivo CSV já existe
if [ -f "/tmp/${now}.csv" ]; then
  get_calc "$1"
else
  # Baixa o arquivo CSV do Banco Central
  url="https://www4.bcb.gov.br/Download/fechamento/${now}.csv"
  curl -s -o "/tmp/${now}.csv" "$url"

  # Verifica se o download foi bem-sucedido
  if [ $? -eq 0 ]; then
    get_calc "$1"
  else
    echo "Erro ao baixar o arquivo CSV."
    exit 1
  fi
fi