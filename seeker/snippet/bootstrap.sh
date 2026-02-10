#date: 2026-02-10T17:55:39Z
#url: https://api.github.com/gists/851bccb12e43d08a4ae81de054ae2b2e
#owner: https://api.github.com/users/loadbalance-sudachi-kun

#!/bin/bash
set -euo pipefail

# ============================================================================
# Claude Code Environment - One-liner Bootstrap
# ============================================================================
#
# Mac:
#   curl -fsSL https://gist.githubusercontent.com/loadbalance-sudachi-kun/851bccb12e43d08a4ae81de054ae2b2e/raw/bootstrap.sh | bash
#
# WSL (inside Ubuntu terminal):
#   curl -fsSL https://gist.githubusercontent.com/loadbalance-sudachi-kun/851bccb12e43d08a4ae81de054ae2b2e/raw/bootstrap.sh | bash
#
# ============================================================================

REPO="cloudnative-co/claude-code-config"
INSTALL_DIR="${HOME}/claude-code-config"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${BOLD}==> $*${NC}"; }

# ----------------------------------------------------------------------------
# OS Detection
# ----------------------------------------------------------------------------
detect_os() {
  case "$(uname -s)" in
    Darwin) OS="macos" ;;
    Linux)
      OS="linux"
      grep -qi microsoft /proc/version 2>/dev/null && OS="wsl"
      ;;
    *) error "Unsupported OS: $(uname -s)" ;;
  esac
  info "Platform: ${OS} ($(uname -m))"
}

# ----------------------------------------------------------------------------
# Step 1: Package manager
# ----------------------------------------------------------------------------
ensure_package_manager() {
  step "Step 1/5: Package manager"

  case "$OS" in
    macos)
      if ! command -v brew &>/dev/null; then
        info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add brew to PATH for Apple Silicon
        if [[ -f /opt/homebrew/bin/brew ]]; then
          eval "$(/opt/homebrew/bin/brew shellenv)"
          # Persist to shell profile
          SHELL_PROFILE="${HOME}/.zprofile"
          if ! grep -q 'brew shellenv' "$SHELL_PROFILE" 2>/dev/null; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$SHELL_PROFILE"
          fi
        fi
        ok "Homebrew installed"
      else
        ok "Homebrew already installed"
      fi
      ;;
    linux|wsl)
      if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        ok "apt-get ready"
      elif command -v dnf &>/dev/null; then
        ok "dnf ready"
      else
        error "No supported package manager found (apt-get or dnf)"
      fi
      ;;
  esac
}

# ----------------------------------------------------------------------------
# Step 2: Core dependencies
# ----------------------------------------------------------------------------
install_dependencies() {
  step "Step 2/5: Core dependencies (git, node, jq, gh, tmux)"

  case "$OS" in
    macos)
      local pkgs=()
      command -v git   &>/dev/null || pkgs+=(git)
      command -v node  &>/dev/null || pkgs+=(node)
      command -v jq    &>/dev/null || pkgs+=(jq)
      command -v gh    &>/dev/null || pkgs+=(gh)
      command -v tmux  &>/dev/null || pkgs+=(tmux)

      if [[ ${#pkgs[@]} -gt 0 ]]; then
        info "Installing: ${pkgs[*]}"
        brew install "${pkgs[@]}"
      fi
      ;;
    linux|wsl)
      local pkgs=()
      command -v git   &>/dev/null || pkgs+=(git)
      command -v jq    &>/dev/null || pkgs+=(jq)
      command -v tmux  &>/dev/null || pkgs+=(tmux)
      command -v curl  &>/dev/null || pkgs+=(curl)

      if [[ ${#pkgs[@]} -gt 0 ]]; then
        info "Installing: ${pkgs[*]}"
        if command -v apt-get &>/dev/null; then
          sudo apt-get install -y -qq "${pkgs[@]}"
        else
          sudo dnf install -y -q "${pkgs[@]}"
        fi
      fi

      # Node.js via NodeSource
      if ! command -v node &>/dev/null; then
        info "Installing Node.js LTS..."
        curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
        sudo apt-get install -y -qq nodejs
      fi

      # GitHub CLI
      if ! command -v gh &>/dev/null; then
        info "Installing GitHub CLI..."
        if command -v apt-get &>/dev/null; then
          sudo mkdir -p -m 755 /etc/apt/keyrings
          curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
            | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
          echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
            | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
          sudo apt-get update -qq && sudo apt-get install -y -qq gh
        fi
      fi

      # dos2unix (WSL)
      if [[ "$OS" == "wsl" ]] && ! command -v dos2unix &>/dev/null; then
        sudo apt-get install -y -qq dos2unix 2>/dev/null || true
      fi
      ;;
  esac

  ok "git $(git --version | awk '{print $3}')"
  ok "node $(node --version)"
  ok "gh $(gh --version 2>/dev/null | head -1 | awk '{print $3}' || echo 'N/A')"
}

# ----------------------------------------------------------------------------
# Step 3: GitHub authentication
# ----------------------------------------------------------------------------
ensure_gh_auth() {
  step "Step 3/5: GitHub authentication"

  if gh auth status &>/dev/null; then
    ok "Already authenticated with GitHub"
  else
    info "GitHub authentication required for private repo access."
    info "A browser window will open for authentication."
    echo ""
    gh auth login --web --git-protocol https
    ok "GitHub authenticated"
  fi
}

# ----------------------------------------------------------------------------
# Step 4: Clone repo
# ----------------------------------------------------------------------------
clone_repo() {
  step "Step 4/5: Clone configuration repository"

  if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Repository already exists at ${INSTALL_DIR}. Pulling latest..."
    git -C "$INSTALL_DIR" pull --ff-only
    ok "Updated to latest"
  else
    if [[ -d "$INSTALL_DIR" ]]; then
      warn "Directory ${INSTALL_DIR} exists but is not a git repo. Backing up..."
      mv "$INSTALL_DIR" "${INSTALL_DIR}.bak.$(date '+%Y%m%d%H%M%S')"
    fi
    gh repo clone "$REPO" "$INSTALL_DIR"
    ok "Cloned to ${INSTALL_DIR}"
  fi
}

# ----------------------------------------------------------------------------
# Step 5: Run setup
# ----------------------------------------------------------------------------
run_setup() {
  step "Step 5/5: Deploy Claude Code configuration"

  # Install Claude Code CLI
  if ! command -v claude &>/dev/null; then
    info "Installing Claude Code CLI..."
    npm install -g @anthropic-ai/claude-code@latest
    ok "Claude Code CLI installed: $(claude --version 2>/dev/null | head -1)"
  else
    ok "Claude Code CLI already installed: $(claude --version 2>/dev/null | head -1)"
  fi

  # Run the deployment (skip prerequisite checks since we just did them)
  bash "$INSTALL_DIR/setup.sh"
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
main() {
  echo ""
  echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}║   Claude Code Environment Bootstrap          ║${NC}"
  echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
  echo ""

  detect_os
  ensure_package_manager
  install_dependencies
  ensure_gh_auth
  clone_repo
  run_setup

  echo ""
  echo -e "${GREEN}${BOLD}Bootstrap complete!${NC}"
  echo ""
  echo "  Start Claude Code:  claude"
  echo "  Login:              claude login"
  echo "  Config repo:        ${INSTALL_DIR}"
  echo ""
}

main "$@"
