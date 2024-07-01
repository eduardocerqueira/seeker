#date: 2024-07-01T16:55:29Z
#url: https://api.github.com/gists/0f6569fc027798d09cd011157c8e3f4b
#owner: https://api.github.com/users/yagoernandes

#!/bin/bash

# Obtém o último commit de merge
LAST_MERGE_COMMIT=$(git log --merges -n 1 --pretty=format:"%H")

# Obtém todos os commits desde o último merge
COMMITS=$(git log $LAST_MERGE_COMMIT..HEAD --pretty=format:"%s")

# Inicializa os incrementos de versão
MAJOR=0
MINOR=0
PATCH=0

# Flags para evitar contar duplicadamente
FIX_COUNTED=false
FEAT_COUNTED=false
BREAKING_CHANGE_COUNTED=false

# Verifica cada commit para identificar chaves de semantic versioning
for COMMIT in $COMMITS; do
    if [[ ($COMMIT == "fix"* || $COMMIT == "test"* || $COMMIT == "refactor"* || $COMMIT == "docs"* || $COMMIT == "style"*) && $FIX_COUNTED == false ]]; then
        PATCH=$((PATCH + 1))
        FIX_COUNTED=true
    elif [[ $COMMIT == "feat"* && $FEAT_COUNTED == false ]]; then
        MINOR=$((MINOR + 1))
        FEAT_COUNTED=true
    elif [[ $COMMIT == *"BREAKING CHANGE"* && $BREAKING_CHANGE_COUNTED == false ]]; then
        MAJOR=$((MAJOR + 1))
        BREAKING_CHANGE_COUNTED=true
    fi
done

# Obtém a última tag de versão, desconsiderando tudo após o '-'
LAST_VERSION=$(git describe --tags $(git rev-list --tags --max-count=1) | sed 's/-.*//')
# echo "LAST_VERSION: $LAST_VERSION"

# Separa a última versão em componentes major, minor e patch
IFS='.' read -r -a VERSION_PARTS <<<"$LAST_VERSION"
CURRENT_MAJOR=${VERSION_PARTS[0]}
CURRENT_MINOR=${VERSION_PARTS[1]}
CURRENT_PATCH=${VERSION_PARTS[2]}

# Calcula a nova versão
NEW_MAJOR=$((CURRENT_MAJOR + MAJOR))
NEW_MINOR=0
NEW_PATCH=0
if [ $CURRENT_MAJOR == $NEW_MAJOR ]; then
    NEW_MINOR=$((CURRENT_MINOR + MINOR))
    if [ $CURRENT_MINOR == $NEW_MINOR ]; then
        NEW_PATCH=$((CURRENT_PATCH + PATCH))
    fi
fi

# Exibe a nova versão
NEW_VERSION="$NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
echo $NEW_VERSION
