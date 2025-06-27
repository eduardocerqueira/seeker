#date: 2025-06-27T16:37:20Z
#url: https://api.github.com/gists/4e6522185d5c0a44844105fedf054979
#owner: https://api.github.com/users/PabloBispo

#!/bin/bash

# ==============================================================================
#   Script para Gerenciar e Associar Chaves SSH a Repositórios Git
#   Versão Corrigida para funcionar com 'curl | bash'
# ==============================================================================

# Garante que o script pare se um comando falhar
set -e

# Função principal
main() {
    # --- Verificação Inicial ---
    if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        echo "❌ Erro: Este script deve ser executado dentro de um repositório Git."
        return 1
    fi

    echo "🔎 Listando chaves SSH públicas disponíveis em ~/.ssh..."
    echo "======================================================"

    # --- Listar Chaves ---
    local keys_pub
    mapfile -t keys_pub < <(find ~/.ssh -maxdepth 1 -type f -name "*.pub" 2>/dev/null)

    if [ ${#keys_pub[@]} -eq 0 ]; then
        echo "⚠️ Nenhuma chave SSH pública (.pub) encontrada em ~/.ssh."
        echo "   Por favor, gere uma chave com 'ssh-keygen' antes de continuar."
        return 1
    fi

    # --- Seleção da Chave ---
    local i=0
    for key in "${keys_pub[@]}"; do
        echo "  [$i] -> $(basename "$key")"
        i=$((i+1))
    done
    echo "======================================================"

    # CORREÇÃO: Ler diretamente do terminal, ignorando o pipe
    read -p "➡️  Digite o número da chave que deseja usar: " key_index < /dev/tty

    if ! [[ "$key_index" =~ ^[0-9]+$ ]] || [ "$key_index" -ge "${#keys_pub[@]}" ]; then
        echo "❌ Seleção inválida."
        return 1
    fi

    local selected_key_pub="${keys_pub[$key_index]}"
    local selected_key_priv="${selected_key_pub%.pub}"

    if [ ! -f "$selected_key_priv" ]; then
        echo "❌ Erro: A chave privada correspondente (${selected_key_priv}) não foi encontrada."
        return 1
    fi

    echo "✅ Você escolheu: $(basename "$selected_key_priv")"
    echo "======================================================"

    # --- Escolha do Modo de Configuração ---
    echo "Onde você gostaria de associar esta chave?"
    echo "  [1] -> Apenas para este repositório (local, em .git/config)"
    echo "  [2] -> Globalmente para seu usuário (via .bashrc e/ou .ssh/config)"
    
    # CORREÇÃO: Ler diretamente do terminal
    read -p "➡️  Escolha a opção (1 ou 2): " config_choice < /dev/tty

    case $config_choice in
        1)
            configure_repo_local
            ;;
        2)
            configure_global
            ;;
        *)
            echo "❌ Opção inválida."
            return 1
            ;;
    esac

    echo "======================================================"
    echo "🎉 Operação finalizada com sucesso!"
}

# --- Funções de Configuração ---

configure_repo_local() {
    echo "⚙️  Configurando a chave para o repositório atual..."
    git config core.sshCommand "ssh -i \"${selected_key_priv}\" -F /dev/null"
    echo "   Configuração local concluída. O arquivo '.git/config' foi atualizado."
}

configure_global() {
    echo "🌍 Configurando a chave globalmente..."
    local BASHRC_FILE=~/.bashrc
    
    # Adiciona a chave ao ssh-agent no .bashrc para persistência
    local BASHRC_COMMANDS
    BASHRC_COMMANDS=$'\n# Inicia o ssh-agent se não estiver em execução e adiciona a chave SSH\nif [ -z "$SSH_AUTH_SOCK" ] ; then\n  eval `ssh-agent -s`\n  ssh-add "'"${selected_key_priv}"'"\nfi'

    if ! grep -q "ssh-add \"${selected_key_priv}\"" "$BASHRC_FILE"; then
        echo "   Adicionando configuração ao ${BASHRC_FILE}..."
        echo "$BASHRC_COMMANDS" >> "$BASHRC_FILE"
        echo "   Por favor, reinicie seu shell ou execute 'source ${BASHRC_FILE}'."
    else
        echo "   A configuração para esta chave já parece existir em seu ${BASHRC_FILE}."
    fi

    # CORREÇÃO: Ler diretamente do terminal
    read -p "➡️  Deseja criar/atualizar uma entrada no seu ~/.ssh/config para um host? (ex: github.com) (s/N): " add_ssh_config < /dev/tty
    if [[ "$add_ssh_config" =~ ^[sS](im)?$ ]]; then
        # CORREÇÃO: Ler diretamente do terminal
        read -p "   Digite o nome do host (ex: github.com): " hostname < /dev/tty

        local SSH_CONFIG_FILE=~/.ssh/config
        local SSH_CONFIG_ENTRY
        SSH_CONFIG_ENTRY=$'Host '"${hostname}"$'\n  HostName '"${hostname}"$'\n  User git\n  IdentityFile '"${selected_key_priv}"$'\n  IdentitiesOnly yes'

        touch "$SSH_CONFIG_FILE"
        # Remove bloco antigo para este host (se existir) para evitar duplicação
        sed -i.bak "/^Host ${hostname}$/,/^\s*$/d" "$SSH_CONFIG_FILE"
        echo -e "\n${SSH_CONFIG_ENTRY}" >> "$SSH_CONFIG_FILE"
        echo "   Entrada para '${hostname}' foi adicionada/atualizada no ${SSH_CONFIG_FILE}."
    fi
}

# --- Execução ---
main