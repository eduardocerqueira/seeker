#date: 2025-06-13T17:05:37Z
#url: https://api.github.com/gists/e351561456276f5136a6c1c2668f5d7c
#owner: https://api.github.com/users/dosjota

#!/bin/zsh

set -e

echo "🚀 Iniciando..."

# Verificar arquitectura
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
  echo "❌ Este script está pensado para Apple Silicon (arm64)."
  exit 1
fi

# Xcode Command Line Tools
if ! xcode-select -p &>/dev/null; then
  echo "🔧 Instalando Xcode Command Line Tools..."
  xcode-select --install
else
  echo "✅ Xcode Command Line Tools ya instalado."
fi

# Homebrew
if ! command -v brew &>/dev/null; then
  echo "🍺 Instalando Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
  echo "✅ Homebrew ya instalado."
  brew update
fi

# Instalar Oh My Zsh
if [[ ! -d "$HOME/.oh-my-zsh" ]]; then
  echo "🎯 Instalando Oh My Zsh..."
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
else
  echo "✅ Oh My Zsh ya instalado."
fi

# Utilidades básicas
for pkg in git wget curl fzf; do
  if ! brew list $pkg &>/dev/null; then
    echo "🔧 Instalando $pkg..."
    brew install $pkg
  else
    echo "✅ $pkg ya instalado."
  fi
done

# Flutter
if ! command -v flutter &>/dev/null; then
  echo "🦋 Instalando Flutter..."
  brew install --cask flutter
  echo 'export PATH="$PATH:/opt/homebrew/Caskroom/flutter/latest/flutter/bin"' >> ~/.zshrc
  export PATH="$PATH:/opt/homebrew/Caskroom/flutter/latest/flutter/bin"
else
  echo "✅ Flutter ya instalado."
fi

# NVM
if ! brew list nvm &>/dev/null; then
  echo "🔧 Instalando nvm..."
  brew install nvm
  mkdir -p ~/.nvm
  echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.zshrc
  echo '[ -s "/opt/homebrew/opt/nvm/nvm.sh" ] && \. "/opt/homebrew/opt/nvm/nvm.sh"' >> ~/.zshrc
  echo '[ -s "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm" ] && \. "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm"' >> ~/.zshrc
  export NVM_DIR="$HOME/.nvm"
  source /opt/homebrew/opt/nvm/nvm.sh
else
  echo "✅ nvm ya instalado."
  export NVM_DIR="$HOME/.nvm"
  source /opt/homebrew/opt/nvm/nvm.sh
fi

# Node.js
if ! command -v node &>/dev/null; then
  echo "⬇️ Instalando Node.js (LTS)..."
  nvm install --lts
  nvm alias default 'lts/*'
else
  echo "✅ Node.js ya instalado: $(node -v)"
fi

# Global packages
for pkg in typescript ts-node yarn; do
  if ! npm list -g $pkg &>/dev/null; then
    echo "🔧 Instalando paquete global $pkg..."
    npm install -g $pkg
  else
    echo "✅ Paquete global $pkg ya instalado."
  fi
done

# Flutter doctor
echo "🩺 Ejecutando flutter doctor..."
flutter doctor

echo "Listo!!"
echo "ejecuta: source ~/.zshrc para actualizar la terminal"
