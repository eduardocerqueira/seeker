#date: 2025-11-19T17:10:36Z
#url: https://api.github.com/gists/3e7c536fd16c0d5564c17210bb494ec0
#owner: https://api.github.com/users/laporeon

#!/bin/bash

# Requer fonttools (ttx)
command -v ttx >/dev/null 2>&1 || { echo >&2 "fonttools/ttx não instalado. Use: sudo apt install fonttools"; exit 1; }

# Cria pasta para fontes personalizadas
mkdir -p ~/.local/share/fonts

# Loop para processar cada .ttf na pasta atual
for fontfile in *.ttf; do
    echo "Processando: $fontfile"

    # Extrai para .ttx (XML)
    ttx -q "$fontfile"
    ttxfile="${fontfile%.ttf}.ttx"

    # Edita os campos de nome no .ttx
    sed -i \
        -e 's/JetBrainsMono Nerd Font/JetBrains Mono/g' \
        -e 's/JetBrainsMonoNerdFont/JetBrainsMono/g' \
        -e 's/JetBrainsMono NF/JetBrains Mono/g' \
        "$ttxfile"

    # Recompila .ttx em novo .ttf
    ttx -q -m "$fontfile" "$ttxfile"

    # Move nova fonte compilada para pasta de fontes do usuário
    newfont="${fontfile%.ttf}#1.ttf"  # nome que o ttx gera
    mv "$newfont" ~/.local/share/fonts/"$fontfile"

    # Limpa arquivos temporários
    rm "$ttxfile"
done

# Atualiza cache de fontes
fc-cache -f -v

echo "✅ Fontes renomeadas e instaladas como 'JetBrains Mono'."