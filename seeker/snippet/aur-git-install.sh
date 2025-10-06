#date: 2025-10-06T17:11:28Z
#url: https://api.github.com/gists/441ca7a3adff3276af11adb378bff4e7
#owner: https://api.github.com/users/lurepos

set -e

if [ -z "$1" ]; then
  echo "Error: No package name specified."
  echo "Usage: $0 <package-name>"
  exit 1
fi

PACKAGE_NAME="$1"
CLONE_URL="https://github.com/archlinux/aur.git"

echo "==> Installing '$PACKAGE_NAME' from AUR git repository..."

if [ -n "$AUR_BUILD_BASE" ]; then
  BUILD_BASE="$AUR_BUILD_BASE"
else
  CANDIDATES=("/var/tmp" "$HOME/.cache" "$HOME/tmp" "/tmp")
  BUILD_BASE=""
  best_avail=0
  for c in "${CANDIDATES[@]}"; do
    [ -z "$c" ] && continue
    mkdir -p "$c" 2>/dev/null || continue
    avail_kb=$(df --output=avail -k "$c" 2>/dev/null | tail -n1 || echo 0)
    avail_kb=${avail_kb:-0}
    if [ "$avail_kb" -gt "$best_avail" ]; then
      best_avail=$avail_kb
      BUILD_BASE="$c"
    fi
  done
  [ -z "$BUILD_BASE" ] && BUILD_BASE="/tmp"
fi

BUILD_DIR=$(mktemp -d -p "$BUILD_BASE" "aur-build-$PACKAGE_NAME-XXXX")
echo "==> Using temporary build directory: $BUILD_DIR"

trap 'echo "==> Cleaning up..."; rm -rf "$BUILD_DIR"' EXIT

cd "$BUILD_DIR"

echo "==> Cloning repository for '$PACKAGE_NAME'..."
git clone --single-branch --branch "$PACKAGE_NAME" "$CLONE_URL" "$PACKAGE_NAME"

cd "$PACKAGE_NAME"

echo "==> Building and installing '$PACKAGE_NAME'..."
makepkg -si --noconfirm

echo "==> Successfully installed '$PACKAGE_NAME'."
