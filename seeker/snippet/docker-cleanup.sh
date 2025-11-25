#date: 2025-11-25T17:07:45Z
#url: https://api.github.com/gists/cbdc9ba3ba16fbeb1607f803523874ab
#owner: https://api.github.com/users/harkaranbrar7

#!/usr/bin/env bash
# docker-cleanup.sh
# Fully cleans up Docker (containers, images, volumes, networks, builder cache, logs)
# Supports --dry-run, --keep-images, and --help

set -euo pipefail

DRY_RUN=false
KEEP_IMAGES=false

usage() {
  echo "Usage: $0 [--dry-run] [--keep-images] [--help]"
  echo "  --dry-run       Show what would be deleted, and print the commands"
  echo "  --keep-images   Do not delete Docker images"
  echo "  --help          Show this help"
}

# Parse Arguments
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --keep-images) KEEP_IMAGES=true ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $arg"; usage; exit 1 ;;
  esac
done

# Helpers
run() {
  # Print the command to stderr (visible) then run if not dry-run
  echo "+ $*" >&2
  if [ "$DRY_RUN" = false ]; then
    eval "$@"
  fi
}

confirm() {
  if [ "$DRY_RUN" = false ]; then
    echo -e "⚠️  This will REMOVE Docker data (containers, images, volumes, networks, caches, logs)"
    read -r -p "Are you sure? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
      echo "Aborted."
      exit 1
    fi
  else
    echo -e "🧪 DRY-RUN MODE ENABLED"
  fi
}

# Check Docker
if ! docker info &>/dev/null; then
  echo -e "❌ Docker is not running or accessible."
  exit 1
fi

confirm

echo ""
echo "🧹 Starting Docker cleanup ($([ "$DRY_RUN" = true ] && echo DRY-RUN || echo ACTUAL DELETE))..."

# Show commands to be run by enabling xtrace only for destructive sections when not dry-run
enable_xtrace() { [ "$DRY_RUN" = false ] && set -x || true; }
disable_xtrace() { [ "$DRY_RUN" = false ] && set +x || true; }

# Containers
echo ""
echo "🔍 Containers (all):"
run docker ps -a -q

echo "⏹️  Stop running containers:"
run 'docker ps -q | xargs -r docker stop'

echo "🗑️  Remove all containers (stopped):"
# Prefer prune as it’s safer and reports reclaimed space, but keep explicit path for transparency
run 'docker container prune -f'
# Equivalent explicit removal (often noisier):
# run 'docker ps -a -q | xargs -r docker rm -f -v'

# Images
echo ""
echo "🔍 Images (all):"
run docker images -a -q

if [ "$KEEP_IMAGES" = true ]; then
  echo "⛔ Skipping image deletion due to --keep-images"
else
  echo "🗑️  Remove all unused images:"
  # Prefer prune to avoid removing images in use by containers
  run 'docker image prune -a -f'
  # Equivalent explicit removal (more aggressive and can error on in-use images):
  # run 'docker images -a -q | xargs -r docker rmi -f'
fi

# Volumes
echo ""
echo "🔍 Volumes (all local):"
run docker volume ls -q

echo "🗑️  Remove unused volumes (be careful: data loss if volumes are not attached):"
run 'docker volume prune -f'
# Explicit path (can be dangerous if misused):
# run 'docker volume ls -q | xargs -r docker volume rm'

# Networks
echo ""
echo "🔍 Dangling Networks:"
run 'docker network ls --filter "dangling=true" -q'

echo "🗑️  Prune unused networks:"
run 'docker network prune -f'

# Builder cache (classic builder)
echo ""
echo "🔍 Builder Cache:"
if [ "$DRY_RUN" = true ]; then
  echo "Showing current builder cache usage (approx):"
  run 'docker system df --verbose | sed -n "/Build cache/,\$p"'
else
  enable_xtrace
  docker builder prune -a -f
  disable_xtrace
fi

# Buildx cache (if available)
if docker buildx version &>/dev/null; then
  echo ""
  echo "🔍 Buildx Cache:"
  if [ "$DRY_RUN" = true ]; then
    run 'docker buildx du || true'
  else
    enable_xtrace
    docker buildx prune --all --force
    disable_xtrace
  fi
fi

# System prune (covers images, containers, networks, build cache; add --volumes to include volumes)
echo ""
echo "🔍 System Prune with volumes:"
if [ "$DRY_RUN" = true ]; then
  echo "Showing disk usage summary:"
  run docker system df
else
  enable_xtrace
  docker system prune -a -f --volumes
  disable_xtrace
fi

# Docker contexts (remove non-default)
echo ""
echo "🔍 Docker Contexts (excluding 'default'):"
unused_contexts="$(docker context ls -q | grep -v '^default$' || true)"
echo "$unused_contexts"
if [ "$DRY_RUN" = false ] && [ -n "${unused_contexts:-}" ]; then
  enable_xtrace
  for ctx in $unused_contexts; do
    docker context rm "$ctx" || true
  done
  disable_xtrace
fi

# Docker logs (Linux)
echo ""
echo "🔍 Docker logs (/var/lib/docker/containers/*.log):"
if [ -d /var/lib/docker/containers ]; then
  run 'sudo find /var/lib/docker/containers/ -type f -name "*.log"'
  if [ "$DRY_RUN" = false ]; then
    enable_xtrace
    sudo find /var/lib/docker/containers/ -type f -name "*.log" -delete
    disable_xtrace
  fi
else
  echo "Not found or not accessible"
fi

# Docker cache dir
echo ""
echo "🔍 Docker cache directory (/var/cache/docker):"
if [ -d /var/cache/docker ]; then
  echo "/var/cache/docker exists"
  if [ "$DRY_RUN" = false ]; then
    enable_xtrace
    sudo rm -rf /var/cache/docker || true
    disable_xtrace
  fi
else
  echo "No Docker cache found"
fi

# Lando cache
echo ""
echo "🔍 Lando cache (~/.lando/cache):"
if [ -d "${HOME}/.lando/cache" ]; then
  echo "~/.lando/cache exists"
  if [ "$DRY_RUN" = false ]; then
    enable_xtrace
    rm -rf "${HOME}/.lando/cache"
    disable_xtrace
  fi
else
  echo "No Lando cache found"
fi

echo ""
if [ "$DRY_RUN" = true ]; then
  echo "✅ Docker cleanup dry-run complete!"
else
  echo "✅ Docker cleanup actual complete!"
fi
