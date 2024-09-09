#date: 2024-09-09T16:44:08Z
#url: https://api.github.com/gists/b402631ca21689b0100b3c58b076bad6
#owner: https://api.github.com/users/oswaldom-code

#!/bin/bash

# Detectar la ruta del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Asignar la ruta de la llave SSH en base a la ubicación del script
SSH_KEY_PATH="$SCRIPT_DIR/id_ed25519" # ajustar al tipo de algoritmo utilizado

# Verifica si el archivo de la llave existe
if [ ! -f "$SSH_KEY_PATH" ]; then
  echo "Error: La llave SSH no se encontró en $SSH_KEY_PATH"
  exit 1
fi

# Función para ejecutar un comando Git con la llave SSH específica
function git_with_ssh() {
  GIT_SSH_COMMAND="ssh -i $SSH_KEY_PATH" git "$@"
}

# Verifica los argumentos del script
if [ $# -lt 2 ]; then
  echo "Uso: $0 [clone|pull|push] [argumentos adicionales de git]"
  exit 1
fi

# Comando de Git que deseas ejecutar
COMMAND=$1
shift # Desplaza los argumentos para que queden los parámetros de Git

# Ejecuta el comando de Git correspondiente
case $COMMAND in
  clone)
    git_with_ssh clone "$@"
    ;;
  pull)
    git_with_ssh pull "$@"
    ;;
  push)
    git_with_ssh push "$@"
    ;;
  *)
    echo "Comando no soportado: $COMMAND"
    echo "Uso: $0 [clone|pull|push] [argumentos adicionales de git]"
    exit 1
    ;;
esac

