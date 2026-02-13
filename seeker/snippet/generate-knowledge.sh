#date: 2026-02-13T17:27:54Z
#url: https://api.github.com/gists/094991511786c4d0e639524e10fe2317
#owner: https://api.github.com/users/kurtwuckertjr

#!/bin/bash
# Generate markdown knowledge base from cloned repos
# Run this on Clurt's VM after clone-all-repos.sh

set -e

REPOS_DIR="$HOME/.openclaw/workspace/repos"
KNOWLEDGE_DIR="$HOME/.openclaw/workspace/knowledge"
mkdir -p "$KNOWLEDGE_DIR"

generate_summary() {
  local repo_path="$1"
  local repo_name="$(basename "$repo_path")"
  local category="$(basename "$(dirname "$repo_path")")"
  local out="$KNOWLEDGE_DIR/${category}--${repo_name}.md"

  echo "  Generating: $category/$repo_name"

  {
    echo "# $repo_name"
    echo ""
    echo "**Category:** $category"
    echo "**Path:** $repo_path"
    echo ""

    # README
    if [ -f "$repo_path/README.md" ]; then
      echo "## README"
      echo ""
      head -100 "$repo_path/README.md"
      echo ""
      echo "---"
      echo ""
    fi

    # Package info
    if [ -f "$repo_path/package.json" ]; then
      echo "## Package Info (package.json)"
      echo ""
      echo '```json'
      # Extract name, version, description, dependencies, scripts
      bun -e "
        const pkg = JSON.parse(require('fs').readFileSync('$repo_path/package.json', 'utf8'));
        const summary = {
          name: pkg.name,
          version: pkg.version,
          description: pkg.description,
          main: pkg.main || pkg.module || pkg.exports,
          scripts: pkg.scripts ? Object.keys(pkg.scripts) : [],
          dependencies: pkg.dependencies ? Object.keys(pkg.dependencies) : [],
          devDependencies: pkg.devDependencies ? Object.keys(pkg.devDependencies) : [],
        };
        console.log(JSON.stringify(summary, null, 2));
      " 2>/dev/null || cat "$repo_path/package.json"
      echo '```'
      echo ""
    fi

    # Go module info
    if [ -f "$repo_path/go.mod" ]; then
      echo "## Go Module (go.mod)"
      echo ""
      echo '```'
      head -30 "$repo_path/go.mod"
      echo '```'
      echo ""
    fi

    # Directory structure (top 2 levels)
    echo "## Directory Structure"
    echo ""
    echo '```'
    find "$repo_path" -maxdepth 2 -not -path '*/node_modules/*' -not -path '*/.git/*' -not -path '*/vendor/*' -not -path '*/.next/*' -not -path '*/dist/*' | sed "s|$repo_path/||" | sort | head -60
    echo '```'
    echo ""

    # Key exports (TypeScript)
    if [ -f "$repo_path/src/index.ts" ] || [ -f "$repo_path/src/index.tsx" ]; then
      echo "## Key Exports"
      echo ""
      echo '```typescript'
      local index_file="$repo_path/src/index.ts"
      [ -f "$repo_path/src/index.tsx" ] && index_file="$repo_path/src/index.tsx"
      grep -E "^export" "$index_file" 2>/dev/null | head -30 || true
      echo '```'
      echo ""
    fi

    # Key exports (Go)
    if ls "$repo_path"/*.go 1>/dev/null 2>&1; then
      echo "## Key Go Files"
      echo ""
      echo '```'
      find "$repo_path" -maxdepth 2 -name "*.go" -not -path '*/vendor/*' | sed "s|$repo_path/||" | sort | head -20
      echo '```'
      echo ""
    fi

    # Test patterns
    local test_count=0
    test_count=$(find "$repo_path" -name "*test*" -o -name "*spec*" -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null | wc -l | tr -d ' ')
    echo "## Test Info"
    echo ""
    echo "Test files found: $test_count"
    if [ "$test_count" -gt 0 ]; then
      echo ""
      echo '```'
      find "$repo_path" -name "*test*" -o -name "*spec*" -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null | sed "s|$repo_path/||" | sort | head -15
      echo '```'
    fi
    echo ""

    # Language breakdown
    echo "## Language"
    echo ""
    if [ -f "$repo_path/package.json" ]; then
      echo "- TypeScript/JavaScript (Node/Bun)"
    fi
    if [ -f "$repo_path/go.mod" ]; then
      echo "- Go"
    fi
    if [ -f "$repo_path/Cargo.toml" ]; then
      echo "- Rust"
    fi
    if [ -f "$repo_path/requirements.txt" ] || [ -f "$repo_path/pyproject.toml" ]; then
      echo "- Python"
    fi
    echo ""

  } > "$out"
}

echo "=== Generating knowledge base ==="
echo "Source: $REPOS_DIR"
echo "Output: $KNOWLEDGE_DIR"
echo ""

for category_dir in "$REPOS_DIR"/*/; do
  category="$(basename "$category_dir")"
  echo "[$category]"
  for repo_dir in "$category_dir"*/; do
    [ -d "$repo_dir" ] && generate_summary "$repo_dir"
  done
  echo ""
done

# Generate master index
INDEX="$KNOWLEDGE_DIR/_INDEX.md"
{
  echo "# b-open-io Knowledge Base Index"
  echo ""
  echo "Generated: $(date)"
  echo ""
  echo "## Repos by Category"
  echo ""
  for category_dir in "$REPOS_DIR"/*/; do
    category="$(basename "$category_dir")"
    echo "### $category"
    echo ""
    for repo_dir in "$category_dir"*/; do
      if [ -d "$repo_dir" ]; then
        repo_name="$(basename "$repo_dir")"
        echo "- **$repo_name** - \`${category}--${repo_name}.md\`"
      fi
    done
    echo ""
  done
} > "$INDEX"

echo "=== Knowledge base generated ==="
ls "$KNOWLEDGE_DIR" | wc -l | xargs -I{} echo "Total files: {}"
echo "Index: $INDEX"
