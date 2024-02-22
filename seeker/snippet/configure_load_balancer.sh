#date: 2024-02-22T17:00:25Z
#url: https://api.github.com/gists/a19649628422cda4319341b9e98a02a8
#owner: https://api.github.com/users/josenilto

#!/bin/bash

# Verifica se o NGINX está instalado
if ! command -v nginx &> /dev/null
then
    echo "NGINX não encontrado. Instalando..."
    sudo apt update
    sudo apt install nginx -y
fi

# Configuração do arquivo de balanceamento de carga
cat <<EOF | sudo tee /etc/nginx/sites-available/load-balancer
upstream backend {
    server 10.0.0.1:80;
    server 10.0.0.2:80;
    server 10.0.0.3:80;
}

server {
    listen 80;
    server_name backend.com;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Criação de link simbólico para ativar o arquivo de configuração
sudo ln -s /etc/nginx/sites-available/load-balancer /etc/nginx/sites-enabled/

# Testa a sintaxe do arquivo de configuração
sudo nginx -t

# Reinicia o NGINX para aplicar as alterações
sudo systemctl restart nginx

echo "Configuração do balanceamento de carga concluída."
