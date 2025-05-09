#date: 2025-05-09T16:52:12Z
#url: https://api.github.com/gists/8a4c50e7a24dbe93ca57abc41f086fe1
#owner: https://api.github.com/users/dzogrim

#!/usr/bin/env bash

# This script finds macOS Finder aliases in the current directory,
# resolves their targets using `resolve-alias`, and for valid folder targets,
# offers to replace each alias with a Unix symlink to the original folder.

# Ensure resolve-alias is available
if ! command -v resolve-alias &> /dev/null; then
    echo "❌ Error: 'resolve-alias' is not in your PATH."
    echo "Please visit https://github.com/mattieb/resolve-alias and compile it first."
    exit 1
fi

# Known document bundles (used *after* resolving)
BUNDLES_EXTENSIONS="pages app rtfd key numbers"

is_bundle() {
    for ext in $BUNDLES_EXTENSIONS; do
        [[ "$1" == *.$ext ]] && return 0
    done
    return 1
}

# Process all alias files
find . -type f -exec file '{}' + | grep -i "Alias" | cut -d: -f1 | while read -r alias; do
    echo "🔍 Checking: $alias"

    # Try to resolve alias
    target=$(resolve-alias "$alias" 2>/dev/null)

    # Empty or failed resolution
    if [[ -z "$target" || ! -e "$target" ]]; then
        echo "❌ Broken or unresolvable alias: $alias"
        continue
    fi

    # Check for known bundle types (but only if valid)
    if is_bundle "$alias"; then
        echo "⚠️ Alias has bundle-like extension (.$(echo "$alias" | rev | cut -d. -f1 | rev)), but it's a real alias: $alias"
    fi

    # Valid folder alias
    if [[ -d "$target" ]]; then
        echo "📁 Folder alias → $alias"
        echo "    ↪ Target: $target"

        if [[ -e "$alias" && ! -L "$alias" ]]; then
            echo -n "   ➤ Convert this alias to a symlink? [y/N]: " > /dev/tty
            read -r answer < /dev/tty
            if [[ "$answer" =~ ^[Yy]$ ]]; then
                rm -f "$alias"
                ln -s "$target" "$alias"
                echo "✅ Symlink created: $alias → $target"
            else
                echo "⏭️  Skipped."
            fi
        else
            echo "⚠️  Skipping: $alias already a symlink or missing."
        fi

    elif [[ -f "$target" ]]; then
        echo "📄 File alias   → $alias"
        echo "    ↪ Target: $target"

    else
        echo "❓ Unknown target type: $alias → $target"
    fi
done
