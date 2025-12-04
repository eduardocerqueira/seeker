#date: 2025-12-04T16:59:20Z
#url: https://api.github.com/gists/0a7e8802dca9ba06f3743b9d469a1740
#owner: https://api.github.com/users/ed-henrique

#!/bin/bash

dir_downloads="does"
primeiro_ano=1980
ultimo_ano=$(date +%Y)
link_doe_rr="https://www.imprensaoficial.rr.gov.br/app/_visualizar-doe/"

# Cria diretório de arquivos do diário oficial
mkdir -p "${dir_downloads}";

for ano in $(seq "${primeiro_ano}" "${ultimo_ano}"); do
  for mes in {01..12}; do
    for dia in {01..31}; do
      printf "Baixando ${ano}${mes}${dia}.pdf";

      # Baixa o diário oficial
      curl \
      --silent \
      "${link_doe_rr}" \
      -X POST \
      -H 'Content-Type: application/x-www-form-urlencoded' \
      --data-raw "doe=${ano}/${mes}/doe-${ano}${mes}${dia}.pdf&ipconexao=" \
      -o "${dir_downloads}/${ano}${mes}${dia}.pdf";

      echo " OK";

      sleep 1;
    done
  done
done
