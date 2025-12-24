#date: 2025-12-24T17:03:57Z
#url: https://api.github.com/gists/6c2408c51d065ea093300253ac649bff
#owner: https://api.github.com/users/thisguymartin

#!/bin/bash

# Development Environment Cleanup Script
# Removes caches, orphaned dependencies, and reclaims disk space

set -e

echo "🧹 Dev Environment Cleanup Starting..."
echo "======================================="

# Track space before
SPACE_BEFORE=$(df -h ~ | awk 'NR==2 {print $4}')
echo "📊 Free space before: $SPACE_BEFORE"

# --- NPM Cleanup ---
if command -v npm &> /dev/null; then
    echo "\n📦 NPM Cleanup..."

    # Clear npm cache
    echo "  → Clearing npm cache..."
    npm cache clean --force 2>/dev/null || true

    # Remove global unused packages (careful - just shows what could be removed)
    echo "  → Checking for extraneous global packages..."
    npm ls -g --depth=0 2>/dev/null || true
fi

# --- Bun Cleanup ---
if command -v bun &> /dev/null; then
    echo "\n🥟 Bun Cleanup..."

    # Clear bun cache
    BUN_CACHE="${HOME}/.bun/install/cache"
    if [ -d "$BUN_CACHE" ]; then
        echo "  → Clearing bun install cache..."
        rm -rf "$BUN_CACHE"/*
    fi

    # Clear bun module cache
    BUN_MODULE_CACHE="${HOME}/.bun/install/cache"
    if [ -d "$BUN_MODULE_CACHE" ]; then
        echo "  → Bun cache cleared"
    fi
fi

# --- pnpm Cleanup ---
if command -v pnpm &> /dev/null; then
    echo "\n📦 pnpm Cleanup..."
    pnpm store prune 2>/dev/null || true
fi

# --- Yarn Cleanup ---
if command -v yarn &> /dev/null; then
    echo "\n🧶 Yarn Cleanup..."
    yarn cache clean 2>/dev/null || true
fi

# --- Go Cleanup ---
if command -v go &> /dev/null; then
    echo "\n🐹 Go Cleanup..."

    # Clean module cache
    echo "  → Cleaning go module cache..."
    go clean -modcache 2>/dev/null || true

    # Clean build cache
    echo "  → Cleaning go build cache..."
    go clean -cache 2>/dev/null || true
fi

# --- Docker Cleanup (optional - commented out for safety) ---
if command -v docker &> /dev/null; then
    echo "\n🐳 Docker Cleanup..."
    echo "  → Removing dangling images and build cache..."
    docker system prune -f 2>/dev/null || true

    # Uncomment for more aggressive cleanup:
    # docker system prune -a -f --volumes
fi

# --- Remove common junk files from home directory ---
echo "\n🗑️  Removing common junk files..."

# DS_Store files
find ~ -name ".DS_Store" -type f -delete 2>/dev/null || true

# Thumbnail caches (macOS)
find ~ -name "Thumbs.db" -type f -delete 2>/dev/null || true

# Track space after
SPACE_AFTER=$(df -h ~ | awk 'NR==2 {print $4}')

echo "\n======================================="
echo "✅ Cleanup complete!"
echo "📊 Free space before: $SPACE_BEFORE"
echo "📊 Free space after:  $SPACE_AFTER"
echo "======================================="

# Optional: Suggest node_modules cleanup
echo "\n💡 Tip: To remove old node_modules from inactive projects, run:"
echo '   find ~/Projects -name "node_modules" -type d -mtime +30 -prune -exec rm -rf {} +'
echo "   (This removes node_modules not modified in 30+ days)"
