#date: 2026-01-06T17:17:08Z
#url: https://api.github.com/gists/0e8687bc7587912bea8538b834503210
#owner: https://api.github.com/users/nerun

#!/bin/bash
# https://xmirror.voidlinux.org/

# Lista de URLs
URLs=(
    "https://mirrors.cicku.me/voidlinux"
    "https://mirror.ps.kz/voidlinux/"
    "https://mirror.nju.edu.cn/voidlinux/"
    "https://mirrors.bfsu.edu.cn/voidlinux/"
    "https://mirrors.tuna.tsinghua.edu.cn/voidlinux/"
    "https://mirror.sjtu.edu.cn/voidlinux/"
    "https://mirrors.kubarcloud.net/voidlinux/"
    "https://repo.jing.rocks/voidlinux/"
    "https://mirror.meowsmp.net/voidlinux/"
    "http://ftp.dk.xemacs.org/voidlinux/"
    "https://mirrors.dotsrc.org/voidlinux/"
    "https://ftp.cc.uoc.gr/mirrors/linux/voidlinux/"
    "https://voidlinux.mirror.garr.it/"
    "https://void.cijber.net/"
    "https://void.sakamoto.pl/"
    "http://ftp.debian.ru/mirrors/voidlinux/"
    "https://mirror.yandex.ru/mirrors/voidlinux/"
    "https://ftp.lysator.liu.se/pub/voidlinux/"
    "https://mirror.accum.se/mirror/voidlinux/"
    "https://mirror.puzzle.ch/voidlinux/"
    "https://mirror.vofr.net/voidlinux/"
    "https://mirror.clarkson.edu/voidlinux/"
    "https://mirrors.lug.mtu.edu/voidlinux/"
    "https://mirror.aarnet.edu.au/pub/voidlinux/"
    "https://ftp.swin.edu.au/voidlinux/"
    "http://void.chililinux.com/voidlinux/"
    "https://mirror.linux.ec/voidlinux/"
    "https://mirror.freedif.org/voidlinux/"
    "https://repo-fi.voidlinux.org/"
    "https://repo-de.voidlinux.org/"
    "https://repo-fastly.voidlinux.org/"
    "https://mirrors.summithq.com/voidlinux/"
)

# Arquivo temporário para armazenar resultados
temp_file=$(mktemp)

echo "Testando ping para cada servidor (3 tentativas cada)..."
echo "======================================================"

# Para cada URL, extrair o domínio e fazer ping
for url in "${URLs[@]}"; do
    # Extrair o domínio da URL
    domain=$(echo "$url" | sed -e 's|^[^/]*//||' -e 's|/.*$||' -e 's|^http://||' -e 's|^https://||')
    
    echo -n "Testando $domain... "
    
    # Fazer ping 3 vezes e capturar o tempo médio
    # -c 3: 3 tentativas
    # -W 2: timeout de 2 segundos
    # grep 'min/avg/max': linha com estatísticas
    # awk: extrair a média
    ping_result=$(ping -c 3 -W 2 "$domain" 2>/dev/null | grep 'min/avg/max')
    
    if [ $? -eq 0 ]; then
        avg_ping=$(echo "$ping_result" | awk -F'/' '{print $5}')
        echo "$avg_ping ms"
        echo "$avg_ping $domain" >> "$temp_file"
    else
        echo "FALHOU ou TIMEOUT"
        echo "99999 $domain" >> "$temp_file"  # Valor alto para falhas
    fi
done

echo ""
echo "======================================================"
echo "TOP 5 servidores com menor ping:"
echo "======================================================"

# Ordenar por ping (menor primeiro) e pegar os 5 primeiros
sort -n "$temp_file" | head -5 | while read ping domain; do
    if [ "$ping" == "99999" ]; then
        echo "$domain: FALHOU"
    else
        printf "%-40s: %.2f ms\n" "$domain" "$ping"
    fi
done

# Limpar arquivo temporário
rm "$temp_file"