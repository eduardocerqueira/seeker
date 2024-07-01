#date: 2024-07-01T16:54:44Z
#url: https://api.github.com/gists/679cc7cd998a5a20b56ba1f625233d7c
#owner: https://api.github.com/users/yagoernandes

#!/bin/bash

# Gera a nova versão
NEW_VERSION=$(./commands/generate_version.sh)

# Obtém o último commit de merge
LAST_MERGE_COMMIT=$(git log --merges -n 1 --pretty=format:"%H")

# Obtém todas as mensagens de commits desde o último merge
COMMITS=$(git log $LAST_MERGE_COMMIT..HEAD --pretty=format:"%s" --no-merges)

# Inicializa arrays para diferentes tipos de commit
FEAT=()
FIX=()
DOCS=()
STYLE=()
REFACTOR=()
TEST=()
CHORE=()

# Função para extrair e formatar a mensagem do commit
extract_commit_message() {
    local PREFIX=$1
    local COMMIT=$2
    local REGEX="$PREFIX\(([^)]+)\):\ (.*)"
    if [[ $COMMIT =~ $REGEX ]]; then
        echo "(${BASH_REMATCH[1]}) ${BASH_REMATCH[2]}"
    else
        echo "${COMMIT#$PREFIX: }"
    fi
}

# Função para formatar a mensagem do commit
format_commit_message() {
    local COMMIT=$1
    case $COMMIT in
        feat*)
            extract_commit_message "feat" "$COMMIT"
            ;;
        fix*)
            extract_commit_message "fix" "$COMMIT"
            ;;
        docs*)
            extract_commit_message "docs" "$COMMIT"
            ;;
        style*)
            extract_commit_message "style" "$COMMIT"
            ;;
        refactor*)
            extract_commit_message "refactor" "$COMMIT"
            ;;
        test*)
            extract_commit_message "test" "$COMMIT"
            ;;
        chore*)
            extract_commit_message "chore" "$COMMIT"
            ;;
    esac
}

# Classifica os commits de acordo com o tipo
while IFS= read -r COMMIT; do
    FORMATTED_COMMIT=$(format_commit_message "$COMMIT")
    case $COMMIT in
        feat*)
            FEAT+=("$FORMATTED_COMMIT")
            ;;
        fix*)
            FIX+=("$FORMATTED_COMMIT")
            ;;
        docs*)
            DOCS+=("$FORMATTED_COMMIT")
            ;;
        style*)
            STYLE+=("$FORMATTED_COMMIT")
            ;;
        refactor*)
            REFACTOR+=("$FORMATTED_COMMIT")
            ;;
        test*)
            TEST+=("$FORMATTED_COMMIT")
            ;;
        chore*)
            CHORE+=("$FORMATTED_COMMIT")
            ;;
    esac
done <<< "$COMMITS"

# Adiciona cabeçalho no changelog
echo "" >>CHANGELOG.md
echo "## [$NEW_VERSION] - $(date +"%d %b %Y")" >>CHANGELOG.md
echo "" >>CHANGELOG.md

# Função para adicionar commits ao changelog
add_commits_to_changelog() {
    local CATEGORY=$1
    local COMMITS=("${!2}")
    if [ ${#COMMITS[@]} -ne 0 ]; then
        echo "### $CATEGORY" >>CHANGELOG.md
        echo "" >>CHANGELOG.md
        for COMMIT in "${COMMITS[@]}"; do
            echo "- $COMMIT" >>CHANGELOG.md
        done
        echo "" >>CHANGELOG.md
    fi
}

# Adiciona os commits classificados no changelog
add_commits_to_changelog "Features" FEAT[@]
add_commits_to_changelog "Fixes" FIX[@]
add_commits_to_changelog "Documentation" DOCS[@]
add_commits_to_changelog "Styling" STYLE[@]
add_commits_to_changelog "Refactoring" REFACTOR[@]
add_commits_to_changelog "Tests" TEST[@]
add_commits_to_changelog "Chores" CHORE[@]

echo "Changelog atualizado com sucesso!"
