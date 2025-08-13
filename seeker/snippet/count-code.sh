#date: 2025-08-13T17:10:55Z
#url: https://api.github.com/gists/9d0a5191523a5ee0a763ce0e42a33d9c
#owner: https://api.github.com/users/marianob-span1

#!/bin/bash

# Script to count files and lines of code in the codebase
# Respects .gitignore and excludes common non-code files

echo "Counting files and lines of code..."
echo "======================================"

# Use git ls-files to respect .gitignore and only count tracked files
# Exclude common non-code extensions
TOTAL_FILES=$(git ls-files | grep -v -E '\.(lock|json\.lock|yarn\.lock|package-lock\.json|log|tmp|cache|DS_Store|env|env\.local|env\.development|env\.production|env\.test)$' | grep -v -E '\.(png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|dmg|exe|dll|so|dylib)$' | wc -l)

# Count lines of code for source files only
TOTAL_LINES=$(git ls-files | grep -E '\.(js|jsx|ts|tsx|py|java|c|cpp|h|hpp|cs|php|rb|go|rs|kt|swift|scala|clj|cljs|elm|hs|ml|fs|pas|pl|sh|bash|zsh|fish|sql|html|htm|css|scss|sass|less|styl|vue|svelte|md|txt|yaml|yml|toml|ini|conf|config|dockerfile|makefile)$' | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')

echo "Total files (excluding common non-code files): $TOTAL_FILES"
echo "Total lines of code: $TOTAL_LINES"
echo ""

# Show breakdown by file type
echo "Breakdown by file type:"
echo "-------------------------"
git ls-files | grep -E '\.(js|jsx|ts|tsx|py|java|c|cpp|h|hpp|cs|php|rb|go|rs|kt|swift|scala|clj|cljs|elm|hs|ml|fs|pas|pl|sh|bash|zsh|fish|sql|html|htm|css|scss|sass|less|styl|vue|svelte|md|txt|yaml|yml|toml|ini|conf|config|dockerfile|makefile)$' | sed 's/.*\.//' | sort | uniq -c | sort -nr

echo ""
echo "Analysis complete!"
