#date: 2025-03-05T17:02:11Z
#url: https://api.github.com/gists/0c2ebe94c476fa07f08d3d24ee8e8b61
#owner: https://api.github.com/users/cairoapcampos

#!/bin/bash

######################################################
# Script de Configuração do WinBox 4.0beta17         #
# Versão: 1.0                                        #
# Autor: Cairo Ap. Campos                            #
######################################################

# Função de configuração do WinBox
Config() {
    echo
    echo "Baixando WinBox..."
    sleep 3
    echo
    wget https://download.mikrotik.com/routeros/winbox/4.0beta17/WinBox_Linux.zip

    echo
    echo "Criando diretório /opt/winbox/..."
    sleep 3
    mkdir -p /opt/winbox/

    echo
    echo "Extraindo arquivos para /opt/winbox/..."
    sleep 3
    echo
    unzip WinBox_Linux.zip -d /opt/winbox/
    
    echo
    echo "Removendo arquivo ZIP..."
    sleep 3
    rm WinBox_Linux.zip

    echo
    echo "Criando atalho no menu..."
    sleep 3
    cat <<EOF > /usr/share/applications/winbox.desktop
[Desktop Entry]
Version=1.0
Type=Application
Name=Winbox
Comment=Winbox
Exec=/opt/winbox/WinBox
Icon=/opt/winbox/assets/img/winbox.png
Terminal=false
Categories=Network;
EOF

    chmod 644 /usr/share/applications/winbox.desktop
    
    echo
    echo "Atualizando base de atalhos..."
    sleep 3
    update-desktop-database /usr/share/applications/
    echo
}

clear

# Verificando se os pacotes estão instalados
if ! command -v wget &> /dev/null; then
    echo
    echo "O pacote Wget não está instalado! Instalando..."
    sleep 3
    echo
    apt update && apt install -y wget
fi

if ! command -v unzip &> /dev/null; then
    echo
    echo "O pacote Unzip não está instalado! Instalando..."
    sleep 3
    echo
    apt update && apt install -y unzip
fi

# Executa a configuração do WinBox
Config
