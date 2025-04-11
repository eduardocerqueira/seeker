#date: 2025-04-11T16:47:47Z
#url: https://api.github.com/gists/35538d4dfb9c6baf297cc7b2096ce8e7
#owner: https://api.github.com/users/brahimmachkouri

#!/bin/bash

set -e

# Vérifie si sudo est disponible
if ! command -v sudo &>/dev/null; then
    echo "❌ sudo n'est pas installé. Abandon."
    exit 1
fi

# Demande le mot de passe sudo dès le début
echo "🔐 Vérification des privilèges sudo..."
sudo -v

# Maintient sudo actif pendant toute l'exécution du script
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

echo "🔧 Mise à jour des paquets..."
sudo apt update && sudo apt install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

echo "🔐 Ajout de la clé GPG Docker..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo "📦 Ajout du dépôt Docker..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "🔄 Mise à jour et installation de Docker..."
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "🧪 Test de Docker..."
sudo docker run --rm hello-world

echo "⚙️ Configuration du daemon Docker avec overlay2..."
sudo mkdir -p /etc/docker
echo '{
  "storage-driver": "overlay2"
}' | sudo tee /etc/docker/daemon.json > /dev/null

echo "🔄 Redémarrage du service Docker..."
sudo systemctl daemon-reexec
sudo systemctl restart docker

echo "👤 Ajout de l'utilisateur courant au groupe docker (si ce n’est pas déjà fait)..."
if groups $USER | grep -qv '\bdocker\b'; then
  sudo usermod -aG docker $USER
  echo "✅ Utilisateur ajouté au groupe docker. Déconnecte-toi/reconnecte-toi pour appliquer."
else
  echo "ℹ️ Utilisateur déjà dans le groupe docker."
fi

echo "✅ Installation et configuration terminées."