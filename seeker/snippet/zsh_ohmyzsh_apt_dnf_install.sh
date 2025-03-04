#date: 2025-03-04T17:10:53Z
#url: https://api.github.com/gists/9726e42599723179fa513d8c01bc7e71
#owner: https://api.github.com/users/tomasmetal23

#!/bin/bash

# Detectar el gestor de paquetes (apt o dnf)
if command -v apt &> /dev/null; then
    PKG_MANAGER="apt"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
else
    echo "No se pudo detectar el gestor de paquetes (apt o dnf). Este script solo funciona en distribuciones basadas en Debian o Fedora."
    exit 1
fi

# Actualizar repositorios
echo "Actualizando repositorios..."
if [[ $PKG_MANAGER == "apt" ]]; then
    sudo apt update
elif [[ $PKG_MANAGER == "dnf" ]]; then
    sudo dnf makecache
fi

# Instalar zsh, git y chsh (si no están instalados)
echo "Instalando zsh, git y chsh..."
if [[ $PKG_MANAGER == "apt" ]]; then
    sudo apt install zsh git util-linux -y
elif [[ $PKG_MANAGER == "dnf" ]]; then
    sudo dnf install zsh git -y
fi

# Cambiar la shell por defecto a zsh (usando chsh si está disponible)
echo "Configurando zsh como shell por defecto..."
if command -v chsh &> /dev/null; then
    chsh -s $(which zsh)
else
    echo "El comando 'chsh' no está disponible. Cambiando la shell manualmente..."
    sudo usermod -s $(which zsh) $(whoami)
    echo "Shell cambiada manualmente a zsh. Por favor, reinicia tu sesión para aplicar los cambios."
fi

# Forzar el cambio a zsh antes de instalar Oh My Zsh
echo "Cambiando a zsh..."
if [[ $SHELL != *"zsh"* ]]; then
    # Si no estamos en zsh, ejecutar el resto del script en zsh
    exec zsh -c '
    # Instalar Oh My Zsh (de manera no interactiva)
    echo "Instalando Oh My Zsh..."
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

    # Instalar plugins populares
    echo "Instalando plugins para zsh..."

    # zsh-autosuggestions
    echo "Instalando zsh-autosuggestions..."
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

    # zsh-syntax-highlighting
    echo "Instalando zsh-syntax-highlighting..."
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

    # zsh-history-substring-search
    echo "Instalando zsh-history-substring-search..."
    git clone https://github.com/zsh-users/zsh-history-substring-search ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search

    # Configurar los plugins en el archivo .zshrc
    echo "Configurando plugins en ~/.zshrc..."
    sed -i "s/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting zsh-history-substring-search)/" ~/.zshrc

    # Recargar la configuración de zsh
    echo "Recargando configuración de zsh..."
    source ~/.zshrc

    echo "¡Instalación completada! Por favor, reinicia tu terminal o ejecuta 'zsh' para empezar a usarlo."
    '
else
    echo "Ya estás en zsh. Continuando con la instalación..."
fi