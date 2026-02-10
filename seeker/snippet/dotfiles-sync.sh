#date: 2026-02-10T17:38:59Z
#url: https://api.github.com/gists/36092f30ea1ec7c454fdaf0d012881c4
#owner: https://api.github.com/users/isdrk

#!/bin/bash
# Dotfiles Sync - Backup/restore via GitHub Gist

set -e

BACKUP_DIR="${HOME}/.dotfiles-backup"
GIST_ID_FILE="${HOME}/.dotfiles-gist-id"

DOTFILES=(
    ".zshrc"
    ".bashrc"
    ".bash_profile"
    ".vimrc"
    ".tmux.conf"
    ".gitconfig"
)

CONFIG_DIRS=(
    ".config/nvim"
)

usage() {
    cat <<EOF
Dotfiles Manager

Commands:
    setup            Setup GitHub token
    backup           Upload to secret Gist
    download <id>    Download to ./dotfiles/
    diff <id>        Show colored diff
    restore <id>     Restore to ~/
    list             Show files

Workflow:
    $0 download abc123
    $0 diff abc123
    $0 restore abc123
EOF
    exit 1
}

setup() {
    echo "Token: "**********"://github.com/settings/tokens/new (gist)"
    read -sp "Paste: "**********"
    echo ""
    [ -z "$token" ] && exit 1

    if curl -sf -H "Authorization: "**********"://api.github.com/gists >/dev/null; then
        echo "$token" > "${HOME}/.github-gist-token"
        chmod 600 "${HOME}/.github-gist-token"
        echo "✓ Saved!"
    else
        echo "✗ Invalid"
        exit 1
    fi
}

backup() {
    [ ! -f "${HOME}/.github-gist-token" ] && echo "Run: "**********"
    token= "**********"

    rm -rf "$BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"

    count=0
    for file in "${DOTFILES[@]}"; do
        [ -e "${HOME}/$file" ] && {
            mkdir -p "$(dirname "$BACKUP_DIR/$file")"
            cp -p "${HOME}/$file" "$BACKUP_DIR/$file"
            echo "  ✓ $file"
            ((count++))
        }
    done

    for dir in "${CONFIG_DIRS[@]}"; do
        [ -d "${HOME}/$dir" ] && {
            mkdir -p "$(dirname "$BACKUP_DIR/$dir")"
            cp -rp "${HOME}/$dir" "$BACKUP_DIR/$dir"
            echo "  ✓ $dir/"
            ((count++))
        }
    done

    [ $count -eq 0 ] && exit 1

    cd "$BACKUP_DIR"
    tar czf dotfiles.tar.gz .??* * 2>/dev/null || true

    b64=$(base64 < dotfiles.tar.gz | tr -d '\n')

    response=$(curl -sf -X POST \
        -H "Authorization: "**********"
        -H "Content-Type: application/json" \
        -d "{\"description\":\"Dotfiles\",\"public\":false,\"files\":{\"dotfiles.tar.gz\":{\"content\":\"$b64\"}}}" \
        https://api.github.com/gists)

    gist_id=$(echo "$response" | grep -o '"id": *"[^"]*"' | head -1 | cut -d'"' -f4)
    [ -z "$gist_id" ] && exit 1

    echo "$gist_id" > "$GIST_ID_FILE"

    echo ""
    echo "✓ Uploaded!"
    echo "ID: $gist_id"
}

download_gist() {
    gist_id="$1"
    output_dir="${2:-./dotfiles}"

    [ -z "$gist_id" ] && exit 1

    gist_json=$(curl -sf "https://api.github.com/gists/$gist_id")
    raw_url=$(echo "$gist_json" | grep -o '"raw_url": *"[^"]*dotfiles[^"]*"' | head -1 | cut -d'"' -f4)
    [ -z "$raw_url" ] && exit 1

    mkdir -p "$output_dir"
    curl -sfL "$raw_url" -o "$output_dir/dotfiles.tar.gz"

    if file "$output_dir/dotfiles.tar.gz" | grep -q "ASCII text"; then
        if base64 -D </dev/null 2>/dev/null; then
            base64 -D -i "$output_dir/dotfiles.tar.gz" -o "$output_dir/dotfiles_decoded.tar.gz"
        else
            base64 -d "$output_dir/dotfiles.tar.gz" > "$output_dir/dotfiles_decoded.tar.gz"
        fi
        mv "$output_dir/dotfiles_decoded.tar.gz" "$output_dir/dotfiles.tar.gz"
    fi

    cd "$output_dir"
    tar xzf dotfiles.tar.gz
    rm dotfiles.tar.gz

    echo "✓ $output_dir"
}

diff_dotfiles() {
    gist_id="$1"
    [ -z "$gist_id" ] && exit 1

    temp_dir=$(mktemp -d)
    echo "Downloading..."
    download_gist "$gist_id" "$temp_dir" >/dev/null 2>&1

    echo ""

    cd "$temp_dir"
    has_diff=false

    # Check if colored diff is available
    if diff --color=always /dev/null /dev/null 2>/dev/null; then
        DIFF_CMD="diff -u --color=always"
        DIFF_CMD_R="diff -ur --color=always"
    else
        DIFF_CMD="diff -u"
        DIFF_CMD_R="diff -ur"
    fi

    for file in "${DOTFILES[@]}"; do
        if [ -e "$file" ]; then
            if [ -e "${HOME}/$file" ]; then
                if ! diff -q "$file" "${HOME}/$file" >/dev/null 2>&1; then
                    echo "━━━ $file ━━━"
                    $DIFF_CMD "${HOME}/$file" "$file" || true
                    echo ""
                    has_diff=true
                fi
            else
                echo "✚ $file (new)"
                has_diff=true
            fi
        fi
    done

    for dir in "${CONFIG_DIRS[@]}"; do
        if [ -d "$dir" ] && [ -d "${HOME}/$dir" ]; then
            if ! diff -qr "${HOME}/$dir" "$dir" >/dev/null 2>&1; then
                echo "━━━ $dir/ ━━━"
                $DIFF_CMD_R "${HOME}/$dir" "$dir" | head -100 || true
                echo ""
                has_diff=true
            fi
        elif [ -d "$dir" ]; then
            echo "✚ $dir/ (new)"
            has_diff=true
        fi
    done

    rm -rf "$temp_dir"

    [ "$has_diff" = false ] && echo "✓ No differences"
}

restore_gist() {
    gist_id="$1"
    [ -z "$gist_id" ] && [ -f "$GIST_ID_FILE" ] && gist_id=$(cat "$GIST_ID_FILE")
    [ -z "$gist_id" ] && exit 1

    temp_dir=$(mktemp -d)
    download_gist "$gist_id" "$temp_dir" >/dev/null

    cd "$temp_dir"

    for file in "${DOTFILES[@]}"; do
        [ -e "$file" ] && {
            [ -e "${HOME}/$file" ] && cp "${HOME}/$file" "${HOME}/${file}.bak"
            mkdir -p "$(dirname "${HOME}/$file")"
            cp -p "$file" "${HOME}/$file"
            echo "  ✓ $file"
        }
    done

    for dir in "${CONFIG_DIRS[@]}"; do
        [ -d "$dir" ] && {
            mkdir -p "$(dirname "${HOME}/$dir")"
            cp -rp "$dir" "${HOME}/$dir"
            echo "  ✓ $dir/"
        }
    done

    rm -rf "$temp_dir"
    echo "✓ Done!"
}

list_files() {
    for file in "${DOTFILES[@]}"; do
        [ -e "${HOME}/$file" ] && echo "  ✓ $file" || echo "  ✗ $file"
    done
    for dir in "${CONFIG_DIRS[@]}"; do
        [ -d "${HOME}/$dir" ] && echo "  ✓ $dir/" || echo "  ✗ $dir/"
    done
}

case "${1:-}" in
    setup) setup ;;
    backup) backup ;;
    download) download_gist "${2:-}" ;;
    diff) diff_dotfiles "${2:-}" ;;
    restore) restore_gist "${2:-}" ;;
    list) list_files ;;
    *) usage ;;
esac
es ;;
    *) usage ;;
esac
