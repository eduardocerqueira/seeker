#date: 2026-02-20T17:34:01Z
#url: https://api.github.com/gists/4878eef571b7f1a54d64746b272cc427
#owner: https://api.github.com/users/edilsonvilarinho

#!/bin/bash

# Cores para o terminal
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # Sem cor

echo -e "${GREEN}--- Iniciando Configuração Completa (Ubuntu + Workspace + Dev Env) ---${NC}"

# 1. Atualizar o sistema
echo -e "${YELLOW}[1/16] Etapa: Atualizando repositórios e pacotes do sistema...${NC}"
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git

# 2. Criar pasta de Workspace
echo -e "${YELLOW}[2/16] Etapa: Criando o diretório de trabalho ~/workspace para projetos...${NC}"
mkdir -p "$HOME/workspace"

# 3. Instalar Java 8 (OpenJDK 8)
echo -e "${YELLOW}[3/16] Etapa: Instalando OpenJDK 8 e configurando JAVA_HOME...${NC}"
sudo apt install -y openjdk-8-jdk
JAVA_PATH=$(update-java-alternatives -l | grep java-1.8.0-openjdk | awk '{print $3}')

if ! grep -q "JAVA_HOME" "$HOME/.bashrc"; then
    echo -e "\n# Configuração Java 8\nexport JAVA_HOME=$JAVA_PATH\nexport PATH=\$JAVA_HOME/bin:\$PATH" >> "$HOME/.bashrc"
fi

# 4. Instalar Maven
echo -e "${YELLOW}[4/16] Etapa: Instalando Apache Maven e configurando variáveis de ambiente...${NC}"
sudo apt install -y maven
if ! grep -q "M2_HOME" "$HOME/.bashrc"; then
    echo -e "\n# Configuração Maven\nexport M2_HOME=/usr/share/maven\nexport MAVEN_HOME=/usr/share/maven\nexport PATH=\$M2_HOME/bin:\$PATH" >> "$HOME/.bashrc"
fi

# 5. Instalar Node.js 20
echo -e "${YELLOW}[5/16] Etapa: Instalando Node.js 20 (LTS) via repositório NodeSource...${NC}"
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# 6. Configuração do SSH
echo -e "${YELLOW}[6/16] Etapa: Verificando e gerando chaves SSH para autenticação...${NC}"
SSH_KEY_PATH="$HOME/.ssh/id_ed25519"
if [ ! -f "$SSH_KEY_PATH" ]; then
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N ""
    eval "$(ssh-agent -s)"
    ssh-add "$SSH_KEY_PATH"
else
    echo -e "${BLUE}Chave SSH já existente. Pulando geração...${NC}"
fi

# 7. Utilitários
echo -e "${YELLOW}[7/16] Etapa: Instalando utilitários (FileZilla, PuTTY, VPN Gnome)...${NC}"
sudo apt install -y filezilla putty network-manager-openvpn-gnome

# 8. Preparar Flatpak
echo -e "${YELLOW}[8/16] Etapa: Configurando suporte a Flatpak e adicionando repositório Flathub...${NC}"
sudo apt install -y flatpak
sudo flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo

# 9. Baixar Android Studio
echo -e "${YELLOW}[9/16] Etapa: Fazendo download dos binários do Android Studio...${NC}"
wget -cO /tmp/android-studio.tar.gz "https://edgedl.me.gvt1.com/edgedl/android/studio/ide-zips/2020.3.1.25/android-studio-2020.3.1.25-linux.tar.gz"

# 10. Extrair Android Studio
echo -e "${YELLOW}[10/16] Etapa: Extraindo arquivos para ~/workspace/android-studio...${NC}"
sudo tar -xzf /tmp/android-studio.tar.gz -C "$HOME/workspace/"

# 11. Ajustar Permissões
echo -e "${YELLOW}[11/16] Etapa: Ajustando permissões de pasta para o usuário $USER...${NC}"
sudo chown -R $USER:$USER "$HOME/workspace/android-studio/"
chmod +x "$HOME/workspace/android-studio/bin/studio.sh"

# 12. Criar atalho no menu
echo -e "${YELLOW}[12/16] Etapa: Criando atalho no menu de aplicativos (.desktop)...${NC}"
cat <<EOF | sudo tee /usr/share/applications/android-studio.desktop > /dev/null
[Desktop Entry]
Version=1.0
Type=Application
Name=Android Studio
Icon=$HOME/workspace/android-studio/bin/studio.svg
Exec="$HOME/workspace/android-studio/bin/studio.sh" %f
Categories=Development;IDE;
Terminal=false
EOF

# 13. OpenFortiVPN
echo -e "${YELLOW}[13/16] Etapa: Instalando cliente OpenFortiVPN via terminal...${NC}"
sudo apt install openfortivpn -y

# 14. Google Chrome via Flatpak
echo -e "${YELLOW}[14/16] Etapa: Instalando Google Chrome via Flatpak...${NC}"
sudo flatpak install flathub com.google.Chrome -y

# 15. Limpeza
echo -e "${YELLOW}[15/16] Etapa: Removendo pacotes residuais e arquivos temporários...${NC}"
sudo apt autoremove -y
rm /tmp/android-studio.tar.gz

# 16. Resultado
echo -e "-------------------------------------------------------"
echo -e "${GREEN}Configuração finalizada com sucesso, Edilson!${NC}"
echo -e "${BLUE}Dica: Rode 'source ~/.bashrc' para carregar as novas variáveis sem deslogar.${NC}"