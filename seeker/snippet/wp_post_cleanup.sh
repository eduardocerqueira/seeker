#date: 2026-03-11T17:41:12Z
#url: https://api.github.com/gists/5df946574321aedca828ba8de68e26ef
#owner: https://api.github.com/users/ajithrn

#!/bin/bash
# =============================================================================
# WordPress Posts & Media Cleanup Script
# =============================================================================
# Usage:
#   bash _del_old_posts.sh [OPTIONS]
#
# Options:
#   --dry-run         Preview what will be deleted without actually deleting
#   --path=/path/to/your/wp/installation   Set WordPress installation path
#   --batch=50        Number of posts to process per batch (default: 50)
#   --year=2024       Year to target (default: 2024)
#   --months="1 2 3"  Filter by specific months (1-12, space-separated, optional)
#                     If omitted, all months in the year are targeted
#   --skip-backup     Skip the database backup step
#   --log=cleanup.log Log file path (default: cleanup-TIMESTAMP.log)
# =============================================================================
# ============================================================
# USAGE EXAMPLES
# ============================================================
#
# 1. Delete all posts from January 2023:
#       ./wp_post_cleanup.sh --year 2023 --month 01
#
# 2. Delete all posts from the entire year 2022:
#       ./wp_post_cleanup.sh --year 2022
#
# 3. Delete posts from March 2024 (dry run — preview only, no deletion):
#       ./wp_post_cleanup.sh --year 2024 --month 03 --dry-run
#
# 4. Delete posts from June 2021 with verbose logging:
#       ./wp_post_cleanup.sh --year 2021 --month 06 --verbose
#
# 5. Delete posts from December 2020 and log output to a file:
#       ./wp_post_cleanup.sh --year 2020 --month 12 | tee cleanup.log
#
# 6. Run via WP-CLI on a specific WordPress path:
#       ./wp_post_cleanup.sh --year 2023 --month 09 --path=/var/www/html
#
# 7. Cron job — auto-clean posts older than 6 months (add to crontab):
#       0 3 1 * * /path/to/wp_post_cleanup.sh --year $(date -d '6 months ago' +\%Y) --month $(date -d '6 months ago' +\%m)
#
# NOTES:
#   - Always run with --dry-run first to preview which posts will be affected.
#   - The script filters posts by date (year/month) only.
#   - Month must be zero-padded (01–12).
#   - If --month is omitted, all posts from the given year are targeted.
# ============================================================

set -euo pipefail

# ─── Default Configuration ───────────────────────────────────────────────────
DRY_RUN=false
WP_PATH="/path/to/your/wp/installation"
BATCH_SIZE=50
TARGET_YEAR=2024
TARGET_MONTHS=""   # empty = all months; set via --months="1 2 3"
SKIP_BACKUP=false
LOG_FILE="cleanup-$(date +%Y%m%d-%H%M%S).log"
SLEEP_BETWEEN_BATCHES=2    # seconds to pause between batches (prevents timeout)
SLEEP_BETWEEN_DELETES=0.1  # seconds between individual deletes (reduces server load)

# ─── Color Output ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ─── Parse Arguments ─────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --dry-run)       DRY_RUN=true ;;
    --path=*)        WP_PATH="${arg#*=}" ;;
    --batch=*)       BATCH_SIZE="${arg#*=}" ;;
    --year=*)        TARGET_YEAR="${arg#*=}" ;;
    --months=*)      TARGET_MONTHS="${arg#*=}" ;;
    --skip-backup)   SKIP_BACKUP=true ;;
    --log=*)         LOG_FILE="${arg#*=}" ;;
    --help|-h)
      head -20 "$0" | tail -15
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $arg${NC}"
      exit 1
      ;;
  esac
done

# ─── Build WP-CLI Base Command ───────────────────────────────────────────────
WP_CMD="wp --allow-root"
if [ -n "$WP_PATH" ]; then
  WP_CMD="wp --allow-root --path=$WP_PATH"
fi

# ─── Validate & Build Month Filter ───────────────────────────────────────────
MONTH_SQL_CLAUSE=""
MONTH_DISPLAY="All"
if [ -n "$TARGET_MONTHS" ]; then
  for m in $TARGET_MONTHS; do
    if ! [[ "$m" =~ ^[0-9]+$ ]] || [ "$m" -lt 1 ] || [ "$m" -gt 12 ]; then
      echo -e "${RED}✖ Invalid month: $m (must be 1-12)${NC}"
      exit 1
    fi
  done
  MONTH_CSV=$(echo "$TARGET_MONTHS" | tr ' ' ',')
  MONTH_SQL_CLAUSE="AND MONTH(post_date) IN ($MONTH_CSV)"
  MONTH_DISPLAY="$TARGET_MONTHS"
fi

# ─── Helper Functions ────────────────────────────────────────────────────────
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  echo "$msg" >> "$LOG_FILE"
  echo -e "$1"
}

progress_bar() {
  local current=$1
  local total=$2
  local label=$3
  local percent=0

  if [ "$total" -gt 0 ]; then
    percent=$(( current * 100 / total ))
  fi

  local filled=$(( percent / 2 ))
  local empty=$(( 50 - filled ))
  local bar=$(printf '%0.s█' $(seq 1 $filled 2>/dev/null) )
  local space=$(printf '%0.s░' $(seq 1 $empty 2>/dev/null) )

  printf "\r  ${CYAN}%s${NC} [${GREEN}%s${NC}%s] ${BOLD}%d/%d${NC} (%d%%) " \
    "$label" "$bar" "$space" "$current" "$total" "$percent"
}

separator() {
  echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"
}

confirm_action() {
  echo ""
  echo -e "${YELLOW}⚠  Are you sure you want to proceed? (y/N):${NC} "
  read -r response
  if [[ ! "$response" =~ ^[Yy]$ ]]; then
    log "${RED}✖ Aborted by user.${NC}"
    exit 0
  fi
}

check_wp_cli() {
  if ! command -v wp &> /dev/null; then
    log "${RED}✖ WP-CLI is not installed or not in PATH.${NC}"
    exit 1
  fi

  if ! $WP_CMD core is-installed 2>/dev/null; then
    log "${RED}✖ WordPress not found. Check --path or run from the WP root directory.${NC}"
    exit 1
  fi
}

# ─── Pre-flight Checks ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║     WordPress ${TARGET_YEAR} Posts & Media Cleanup Script      ║${NC}"
echo -e "${BOLD}${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

if $DRY_RUN; then
  echo -e "  ${YELLOW}🔍 DRY RUN MODE — Nothing will be deleted${NC}"
  echo ""
fi

log "${BLUE}► Checking WP-CLI installation...${NC}"
check_wp_cli
log "${GREEN}✔ WP-CLI is working.${NC}"

# Check current user (warn if root)
CURRENT_USER=$(whoami)
if [ "$CURRENT_USER" = "root" ]; then
  echo -e "  ${YELLOW}⚠  Running as root! Consider: sudo -u media bash $0 $*${NC}"
  echo ""
fi

# ─── Step 1: Gather Data ────────────────────────────────────────────────────
separator
log "${BLUE}► Step 1: Gathering ${TARGET_YEAR} posts (months: ${MONTH_DISPLAY})...${NC}"

# Use SQL to fetch post IDs — supports year + optional month filtering
POST_QUERY="SELECT ID FROM wp_posts
  WHERE post_type = 'post'
    AND YEAR(post_date) = ${TARGET_YEAR}
    ${MONTH_SQL_CLAUSE}
  ORDER BY ID;"

POST_IDS_RAW=$($WP_CMD db query "$POST_QUERY" --skip-column-names 2>/dev/null || echo "")

if [ -z "$POST_IDS_RAW" ]; then
  log "${YELLOW}⚠  No posts found for year ${TARGET_YEAR} (months: ${MONTH_DISPLAY}). Nothing to do.${NC}"
  exit 0
fi

# Parse result into array (one ID per line)
declare -a POST_IDS=()
while IFS= read -r line; do
  line=$(echo "$line" | tr -d '[:space:]')
  if [ -n "$line" ] && [[ "$line" =~ ^[0-9]+$ ]]; then
    POST_IDS+=("$line")
  fi
done <<< "$POST_IDS_RAW"

TOTAL_POSTS=${#POST_IDS[@]}

if [ "$TOTAL_POSTS" -eq 0 ]; then
  log "${YELLOW}⚠  No posts found for year ${TARGET_YEAR} (months: ${MONTH_DISPLAY}). Nothing to do.${NC}"
  exit 0
fi

log "${GREEN}✔ Found ${BOLD}${TOTAL_POSTS}${NC}${GREEN} posts from ${TARGET_YEAR} (months: ${MONTH_DISPLAY}).${NC}"

# ─── Step 2: Count Attachments (fast SQL scan) ──────────────────────────────
log "${BLUE}► Step 2: Scanning for media attachments (SQL batch mode)...${NC}"

# Build comma-separated list of post IDs for SQL IN clause
POST_IDS_CSV=$(IFS=,; echo "${POST_IDS[*]}")

# Single SQL query: get all directly-attached media + featured images in one shot
ATTACHMENT_SQL="
SELECT DISTINCT id FROM (
  -- Directly attached media (post_parent = any target post)
  SELECT ID AS id FROM wp_posts
  WHERE post_type = 'attachment'
    AND post_parent IN ($POST_IDS_CSV)
  UNION
  -- Featured images via _thumbnail_id meta
  SELECT CAST(meta_value AS UNSIGNED) AS id FROM wp_postmeta
  WHERE post_id IN ($POST_IDS_CSV)
    AND meta_key = '_thumbnail_id'
    AND meta_value REGEXP '^[0-9]+$'
) AS combined
ORDER BY id;
"

log "  Running batch SQL query for attachments..."
ATTACH_RAW=$($WP_CMD db query "$ATTACHMENT_SQL" --skip-column-names 2>/dev/null || echo "")

# Parse result into array (one ID per line)
declare -a UNIQUE_ATTACHMENT_IDS=()
if [ -n "$ATTACH_RAW" ]; then
  while IFS= read -r line; do
    line=$(echo "$line" | tr -d '[:space:]')
    if [ -n "$line" ] && [[ "$line" =~ ^[0-9]+$ ]]; then
      UNIQUE_ATTACHMENT_IDS+=("$line")
    fi
  done <<< "$ATTACH_RAW"
fi

TOTAL_ATTACHMENTS=${#UNIQUE_ATTACHMENT_IDS[@]}

log "${GREEN}✔ Found ${BOLD}${TOTAL_ATTACHMENTS}${NC}${GREEN} media attachments to remove.${NC}"

# ─── Summary & Confirmation ─────────────────────────────────────────────────
separator
echo ""
echo -e "  ${BOLD}Summary:${NC}"
echo -e "  ├── Posts to delete:       ${BOLD}${TOTAL_POSTS}${NC}"
echo -e "  ├── Attachments to delete: ${BOLD}${TOTAL_ATTACHMENTS}${NC}"
echo -e "  ├── Batch size:            ${BOLD}${BATCH_SIZE}${NC}"
echo -e "  ├── Year:                  ${BOLD}${TARGET_YEAR}${NC}"
echo -e "  ├── Months:                ${BOLD}${MONTH_DISPLAY}${NC}"
echo -e "  ├── Log file:              ${BOLD}${LOG_FILE}${NC}"
echo -e "  └── Mode:                  ${BOLD}$(if $DRY_RUN; then echo 'DRY RUN'; else echo 'LIVE DELETE'; fi)${NC}"
echo ""

if ! $DRY_RUN; then
  confirm_action
fi

# ─── Step 3: Database Backup ────────────────────────────────────────────────
if ! $SKIP_BACKUP && ! $DRY_RUN; then
  separator
  BACKUP_FILE="db-backup-before-${TARGET_YEAR}-cleanup-$(date +%Y%m%d-%H%M%S).sql"
  log "${BLUE}► Step 3: Backing up database to ${BACKUP_FILE}...${NC}"
  $WP_CMD db export "$BACKUP_FILE" 2>/dev/null
  BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
  log "${GREEN}✔ Backup complete (${BACKUP_SIZE}).${NC}"
else
  log "${YELLOW}► Step 3: Skipping backup.${NC}"
fi

# ─── Step 4: Delete Attachments (Batched) ───────────────────────────────────
separator
log "${BLUE}► Step 4: Deleting media attachments...${NC}"

DELETED_ATTACHMENTS=0
FAILED_ATTACHMENTS=0
TOTAL_BATCHES_ATT=$(( (TOTAL_ATTACHMENTS + BATCH_SIZE - 1) / BATCH_SIZE ))
CURRENT_BATCH=0

for (( i=0; i<TOTAL_ATTACHMENTS; i+=BATCH_SIZE )); do
  CURRENT_BATCH=$((CURRENT_BATCH + 1))
  BATCH_END=$((i + BATCH_SIZE))
  if [ $BATCH_END -gt $TOTAL_ATTACHMENTS ]; then
    BATCH_END=$TOTAL_ATTACHMENTS
  fi

  BATCH_IDS=("${UNIQUE_ATTACHMENT_IDS[@]:$i:$BATCH_SIZE}")

  log "  ${CYAN}Batch ${CURRENT_BATCH}/${TOTAL_BATCHES_ATT}${NC}" >> "$LOG_FILE"

  for AID in "${BATCH_IDS[@]}"; do
    DELETED_ATTACHMENTS=$((DELETED_ATTACHMENTS + 1))
    progress_bar "$DELETED_ATTACHMENTS" "$TOTAL_ATTACHMENTS" "Media  "

    if ! $DRY_RUN; then
      if $WP_CMD post delete "$AID" --force 2>>"$LOG_FILE"; then
        echo "[OK] Deleted attachment $AID" >> "$LOG_FILE"
      else
        FAILED_ATTACHMENTS=$((FAILED_ATTACHMENTS + 1))
        echo "[FAIL] Could not delete attachment $AID" >> "$LOG_FILE"
      fi
      sleep "$SLEEP_BETWEEN_DELETES"
    fi
  done

  # Pause between batches to prevent server timeout
  if [ $CURRENT_BATCH -lt $TOTAL_BATCHES_ATT ] && ! $DRY_RUN; then
    sleep "$SLEEP_BETWEEN_BATCHES"
  fi
done

echo "" # newline after progress bar
log "${GREEN}✔ Media cleanup done. Deleted: $((DELETED_ATTACHMENTS - FAILED_ATTACHMENTS)), Failed: ${FAILED_ATTACHMENTS}${NC}"

# ─── Step 5: Delete Posts (Batched) ─────────────────────────────────────────
separator
log "${BLUE}► Step 5: Deleting posts...${NC}"

DELETED_POSTS=0
FAILED_POSTS=0
TOTAL_BATCHES_POST=$(( (TOTAL_POSTS + BATCH_SIZE - 1) / BATCH_SIZE ))
CURRENT_BATCH=0

for (( i=0; i<TOTAL_POSTS; i+=BATCH_SIZE )); do
  CURRENT_BATCH=$((CURRENT_BATCH + 1))
  BATCH_END=$((i + BATCH_SIZE))
  if [ $BATCH_END -gt $TOTAL_POSTS ]; then
    BATCH_END=$TOTAL_POSTS
  fi

  BATCH_IDS=("${POST_IDS[@]:$i:$BATCH_SIZE}")

  log "  ${CYAN}Batch ${CURRENT_BATCH}/${TOTAL_BATCHES_POST}${NC}" >> "$LOG_FILE"

  for PID in "${BATCH_IDS[@]}"; do
    DELETED_POSTS=$((DELETED_POSTS + 1))
    progress_bar "$DELETED_POSTS" "$TOTAL_POSTS" "Posts  "

    if ! $DRY_RUN; then
      if $WP_CMD post delete "$PID" --force 2>>"$LOG_FILE"; then
        echo "[OK] Deleted post $PID" >> "$LOG_FILE"
      else
        FAILED_POSTS=$((FAILED_POSTS + 1))
        echo "[FAIL] Could not delete post $PID" >> "$LOG_FILE"
      fi
      sleep "$SLEEP_BETWEEN_DELETES"
    fi
  done

  # Pause between batches
  if [ $CURRENT_BATCH -lt $TOTAL_BATCHES_POST ] && ! $DRY_RUN; then
    sleep "$SLEEP_BETWEEN_BATCHES"
  fi
done

echo "" # newline after progress bar
log "${GREEN}✔ Post cleanup done. Deleted: $((DELETED_POSTS - FAILED_POSTS)), Failed: ${FAILED_POSTS}${NC}"

# ─── Step 6: Flush Caches ──────────────────────────────────────────────────
separator
if ! $DRY_RUN; then
  log "${BLUE}► Step 6: Flushing caches...${NC}"
  $WP_CMD cache flush 2>/dev/null && log "${GREEN}✔ Object cache flushed.${NC}" || log "${YELLOW}⚠  Object cache flush skipped.${NC}"
  $WP_CMD transient delete --all 2>/dev/null && log "${GREEN}✔ Transients cleared.${NC}" || log "${YELLOW}⚠  Transient clear skipped.${NC}"
  $WP_CMD rewrite flush 2>/dev/null && log "${GREEN}✔ Rewrite rules flushed.${NC}" || log "${YELLOW}⚠  Rewrite flush skipped.${NC}"
else
  log "${YELLOW}► Step 6: Skipping cache flush (dry run).${NC}"
fi

# ─── Final Report ───────────────────────────────────────────────────────────
separator
echo ""
echo -e "${BOLD}${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║                   CLEANUP COMPLETE                        ║${NC}"
echo -e "${BOLD}${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Results:${NC}"
echo -e "  ├── Posts deleted:       ${BOLD}$((DELETED_POSTS - FAILED_POSTS))${NC} / ${TOTAL_POSTS}"
echo -e "  ├── Posts failed:        ${BOLD}${FAILED_POSTS}${NC}"
echo -e "  ├── Media deleted:       ${BOLD}$((DELETED_ATTACHMENTS - FAILED_ATTACHMENTS))${NC} / ${TOTAL_ATTACHMENTS}"
echo -e "  ├── Media failed:        ${BOLD}${FAILED_ATTACHMENTS}${NC}"
echo -e "  ├── Mode:                ${BOLD}$(if $DRY_RUN; then echo 'DRY RUN (nothing deleted)'; else echo 'LIVE'; fi)${NC}"
echo -e "  └── Log:                 ${BOLD}${LOG_FILE}${NC}"
echo ""

if $DRY_RUN; then
  echo -e "  ${YELLOW}💡 This was a dry run. Run without --dry-run to actually delete.${NC}"
  echo ""
fi

log "${GREEN}Script finished at $(date).${NC}"