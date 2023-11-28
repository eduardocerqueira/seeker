#date: 2023-11-28T17:00:37Z
#url: https://api.github.com/gists/4da9a75a69443a75c54851e57c4cb80d
#owner: https://api.github.com/users/matheusot

#!/bin/bash

# Nome do arquivo de log
data_hora=$(date "+%Y_%m_%d_%H")
log_file="metricas_${data_hora}.log"

# Coleta métricas e escreve no arquivo de log
{
  date "+%Y-%m-%d %H:%M:%S"
  echo "\n"
  echo "Uso da CPU:"
  top -b -n 1 | grep "%Cpu" | awk '{print "  Usuário: " $2 "%, Sistema: " $4 "%, Ni: " $6 "%, Oci: " $8 "%, O ocioso: " $10 "%"}'
  echo "\n"
  echo "Uso da Memória:"
  free -m | grep Mem | awk '{print "  Total: " $2 " MB, Usado: " $3 " MB, Livre: " $4 " MB"}'
  free -m | grep Swap | awk '{print "Swap Total: " $2 " MB, Swap Usado: " $3 " MB, Swap Livre: " $4 " MB"}'
  echo "\n"
  echo "Uso do Disco:"
  df -h | grep dev |  awk '{print "  " $1 " " $5 " usado"}' | grep -v "0%"
  echo "\n"
  echo "Uso da Rede:"
  /usr/sbin/ifconfig
  vnstat -5 -i wlan0 | head -n 5 | tail -n 2; vnstat -5 -i wlan0 | tail -n 6
  echo "\n"
  echo "Top 10 Processos em Execução:"
  ps aux --sort=-%cpu | head -n 11 | awk 'NR>1 {print " "$11" "$2" CPU:"$3"% Mem:"$4"% "}' | tail -n 10 | column -t
  echo "\n"
  echo "Uptime:"
  uptime
  echo "\n\n"
  echo "=========="
  echo "\n\n"
} >> "$log_file"