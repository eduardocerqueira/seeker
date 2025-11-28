#date: 2025-11-28T17:03:17Z
#url: https://api.github.com/gists/ac5b21051456d2d96ac92b232af9cc4a
#owner: https://api.github.com/users/lardezzoni

#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/errolgr/pd2-trade}"
DIR="${DIR:-$HOME/pd2-trade}"
NODE_MAJOR="${NODE_MAJOR:-22}"

log() { printf "\n\033[1;36m==>\033[0m %s\n" "$*"; }

log "Installing system dependencies (webkit + gstreamer + build tools)..."
sudo apt update
sudo apt install -y \
  build-essential pkg-config curl git ca-certificates \
  libwebkit2gtk-4.1-0 \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
  imagemagick \
  ripgrep

log "Setting up nvm + Node.js ${NODE_MAJOR}..."
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
fi
# shellcheck disable=SC1090
source "$NVM_DIR/nvm.sh"
nvm install "$NODE_MAJOR"
nvm use "$NODE_MAJOR"

log "Cloning repo into: $DIR"
if [[ -d "$DIR/.git" ]]; then
  log "Repo already exists, pulling latest..."
  git -C "$DIR" pull --rebase
else
  git clone "$REPO_URL" "$DIR"
fi

cd "$DIR"

log "Installing JS deps..."
npm install

log "Applying Linux build patches..."
# 1) Ensure missing icon exists (Tauri expects favicon-96x96.png)
if [[ -f "src-tauri/icons/favicon-32x32.png" && ! -f "src-tauri/icons/favicon-96x96.png" ]]; then
  log "Creating src-tauri/icons/favicon-96x96.png from favicon-32x32.png"
  convert "src-tauri/icons/favicon-32x32.png" -resize 96x96 "src-tauri/icons/favicon-96x96.png"
fi

# 2) Split window module: rename Windows file
if [[ -f "src-tauri/src/modules/window.rs" ]]; then
  log "Renaming window.rs -> window_windows.rs"
  mv "src-tauri/src/modules/window.rs" "src-tauri/src/modules/window_windows.rs"
fi

# 3) Create Linux stub
log "Writing src-tauri/src/modules/window_linux.rs"
cat > "src-tauri/src/modules/window_linux.rs" <<'EOF'
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WindowRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

// Linux stub: prevents panics; can be improved later with real monitor/workarea detection.
fn default_rect() -> WindowRect {
    WindowRect { x: 0, y: 0, width: 1280, height: 720 }
}

pub fn get_diablo_rect() -> Option<WindowRect> { None }
pub fn is_diablo_focused() -> bool { false }
pub fn get_work_area() -> Option<WindowRect> { Some(default_rect()) }
pub fn get_appropriate_window_bounds() -> Option<WindowRect> { Some(default_rect()) }
pub fn initialize_foreground_monitoring<F: Fn() + Send + 'static>(_callback: F) {}
pub fn cleanup_foreground_monitoring() {}
EOF

# 4) Fix modules/mod.rs to include all modules + cfg-based window selection
log "Writing src-tauri/src/modules/mod.rs"
cat > "src-tauri/src/modules/mod.rs" <<'EOF'
pub mod commands;
pub mod keyboard;
pub mod system;
pub mod webview;

#[cfg(windows)]
pub mod window_windows;

#[cfg(not(windows))]
pub mod window_linux;

// Keep the public path as `modules::window::*`
#[cfg(windows)]
pub use window_windows as window;

#[cfg(not(windows))]
pub use window_linux as window;
EOF

# 5) Fix Linux crash: avoid app.primary_monitor().unwrap().unwrap()
log "Patching src-tauri/src/lib.rs (avoid unwrap on primary_monitor)..."
python3 - <<'PY'
import re, pathlib, sys
p = pathlib.Path("src-tauri/src/lib.rs")
s = p.read_text(encoding="utf-8")

pattern = re.compile(
    r'#\[cfg\(not\(target_os = "windows"\)\)\]\s*'
    r'let\s*\(x,\s*y,\s*width,\s*height\)\s*=\s*\{\s*'
    r'let\s*monitor\s*=\s*app\.primary_monitor\(\)\.unwrap\(\)\.unwrap\(\);\s*'
    r'let\s*size\s*=\s*monitor\.size\(\);\s*'
    r'let\s*position\s*=\s*monitor\.position\(\);\s*'
    r'\(\s*position\.x\s*as\s*f64,\s*position\.y\s*as\s*f64,\s*'
    r'size\.width\s*as\s*f64,\s*size\.height\s*as\s*f64,\s*\)\s*'
    r'\};',
    re.S
)

replacement = r'''#[cfg(not(target_os = "windows"))]
            let (x, y, width, height) = {
                let
