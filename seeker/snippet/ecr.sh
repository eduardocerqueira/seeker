#date: 2026-02-06T17:30:41Z
#url: https://api.github.com/gists/1a4de129806a8da0ab854908d0e946bb
#owner: https://api.github.com/users/hexed

#!/usr/bin/env bash

# =========================
# USAGE
# =========================
usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Delete all collections from an Emby server.

Options:
  -u, --url URL        Emby server URL (required)
  -k, --key API_KEY    Emby API key (required)
  -t, --threads NUM    Number of parallel workers (default: 6)
  -d, --dry-run        Show what would be deleted without actually deleting
  -h, --help           Show this help message

Examples:
  $0 -u https://emby.example.com -k YOUR_API_KEY --dry-run
  $0 --url https://emby.example.com --key YOUR_API_KEY --threads 10
EOF
  exit 1
}

# =========================
# DEFAULTS
# =========================
THREADS=6
DRY_RUN=0
TMP_FILE="/tmp/emby_collections.txt"

# =========================
# PARSE ARGUMENTS
# =========================
while [[ $# -gt 0 ]]; do
  case $1 in
    -u|--url)
      EMBY_URL="$2"
      shift 2
      ;;
    -k|--key)
      API_KEY="$2"
      shift 2
      ;;
    -t|--threads)
      THREADS="$2"
      shift 2
      ;;
    -d|--dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# =========================
# VALIDATE
# =========================
if [ -z "$EMBY_URL" ] || [ -z "$API_KEY" ]; then
  echo "Error: URL and API key are required"
  echo
  usage
fi

# Remove trailing slash from URL if present
EMBY_URL="${EMBY_URL%/}"

# =========================
# FETCH COLLECTION IDS
# =========================
echo "Fetching collections..."
curl -s \
  -H "X-Emby-Token: "**********"
  "$EMBY_URL/Items?IncludeItemTypes=BoxSet&Recursive=true" \
| jq -r '.Items[] | "\(.Id)\t\(.Name)"' > "$TMP_FILE"

TOTAL=$(wc -l < "$TMP_FILE" | tr -d ' ')
echo "Found $TOTAL collections"

if [ "$TOTAL" -eq 0 ]; then
  echo "Nothing to delete."
  rm -f "$TMP_FILE"
  exit 0
fi

# =========================
# DRY RUN MODE
# =========================
if [ "$DRY_RUN" -eq 1 ]; then
  echo
  echo "🔍 DRY RUN MODE - Collections that would be deleted:"
  echo "=================================================="
  while IFS=$'\t' read -r id name; do
    echo "  [$id] $name"
  done < "$TMP_FILE"
  echo "=================================================="
  echo "Total: $TOTAL collections"
  echo
  echo "Run without --dry-run to actually delete these collections."
  rm -f "$TMP_FILE"
  exit 0
fi

# =========================
# ACTUAL DELETION
# =========================
echo "Deleting collections with $THREADS parallel workers..."
echo

cut -f1 "$TMP_FILE" | xargs -P "$THREADS" -I {} bash -c '
  ID="$1"
  echo "Deleting $ID"
  curl -s \
    --connect-timeout 5 \
    --max-time 20 \
    -X DELETE \
    -H "X-Emby-Token: "**********"
    "'"$EMBY_URL"'/Items/$ID"
' _ {}

echo
echo "✅ All collections deleted."

# =========================
# CLEANUP
# =========================
rm -f "$TMP_FILE"E"