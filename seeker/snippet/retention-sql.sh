#date: 2025-09-17T16:49:24Z
#url: https://api.github.com/gists/2b25bcee19a3c8462e0261b2c0944956
#owner: https://api.github.com/users/Kamesuta

#!/usr/bin/env bash
# retention-sql.sh - バックアップの世代管理（GFS）を行う最小構成スクリプト
# 対象ファイル: /mnt/gdrive/database/sqlbackup-YYYY-MM-DDTHHMMSSZ.sql （UTC想定）
# 使い方:
#   ドライラン:  script/retention-sql.sh
#   削除実行  :  script/retention-sql.sh --apply
#
# 固定の保持ポリシー（変更不可）:
#   - Hourly : 直近 48 時間 → 各「時」に最新 1 件
#   - Daily  : 直近 30 日   → 各「日」に最新 1 件
#   - Weekly : 直近 12 週   → 各「週」に最新 1 件（週は日曜始まり）
#   - Monthly: 直近 24 ヶ月 → 各「月」に最新 1 件
#   - Yearly : 直近 5 年    → 各「年」に最新 1 件
#
# 実装方針（安定性重視）:
#   - ファイル名から時刻（UTC）を解析して epoch（秒）を取得（比較用）
#   - 「どのバケットに属するか」のキーは **ファイル名から直接切り出し**（環境差で壊れにくい）
#   - 直近の範囲判定にのみ epoch を使用（境界の比較が簡単）
#   - デフォルトはドライラン（--apply が **あるときだけ** rm -f 実行）
#
# 注意:
#   - ファイル名が規則に合わないものは対象外（スキップ）
#   - 表示は UTC 時刻で行います

set -euo pipefail

# --- 設定（固定） -------------------------------------------------------------
DIR="/mnt/gdrive/database"     # バックアップディレクトリ（固定）
APPLY="no"                     # 既定はドライラン
[[ $# -eq 1 && "$1" == "--apply" ]] && APPLY="yes"

# --- しきい値の計算（UTCで統一） ----------------------------------------------
# ※ Hour/Day/Week は単純な秒数引きでOK
#    Month/Year は @epoch 相対が壊れる環境があるため "YYYY-MM-DD - N months/years" の形で計算
now_epoch=$(date -u +%s)
cut_hour=$(( now_epoch - 48*3600 ))          # 48 時間前（Hourly 範囲の下限）
cut_day=$((  now_epoch - 30*24*3600 ))       # 30 日前（Daily 範囲の下限）
cut_week=$(( now_epoch - 12*7*24*3600 ))     # 12 週前（Weekly 範囲の下限）
cut_month=$(date -u -d "$(date -u +%Y-%m-%d) -24 months" +%s)  # 24 ヶ月前（Monthly）
cut_year=$(date  -u -d "$(date -u +%Y-%m-%d) -5 years"   +%s)  # 5 年前（Yearly）

# --- ユーティリティ -------------------------------------------------------------
# ファイル名から epoch 秒（UTC）を抽出。失敗時は空文字を返す。
parse_epoch_from_basename() {
  local base="$1"
  # 例: sqlbackup-2025-09-17T153001Z.sql
  if [[ "$base" =~ ^sqlbackup-([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2})([0-9]{2})([0-9]{2})Z\.sql$ ]]; then
    local y=${BASH_REMATCH[1]} m=${BASH_REMATCH[2]} d=${BASH_REMATCH[3]}
    local H=${BASH_REMATCH[4]} M=${BASH_REMATCH[5]} S=${BASH_REMATCH[6]}
    date -u -d "${y}-${m}-${d} ${H}:${M}:${S} UTC" +%s 2>/dev/null || echo ""
  else
    echo ""
  fi
}

# 表示用: epoch → ISO8601(UTC)
fmt_ts_utc() { date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ; }

# 期間キー（すべてファイル名から直接切り出し）
#   base = "sqlbackup-YYYY-MM-DDTHHMMSSZ.sql"
hour_key_from_base()  { local b="$1"; echo "${b:10:13}"; } # "YYYY-MM-DDTHH"
day_key_from_base()   { local b="$1"; echo "${b:10:10}"; } # "YYYY-MM-DD"
month_key_from_base() { local b="$1"; echo "${b:10:7}";  } # "YYYY-MM"
year_key_from_base()  { local b="$1"; echo "${b:10:4}";  } # "YYYY"

# 週キー（日曜始まり）: その週の日曜の日付 "YYYY-MM-DD"
week_key_from_base() {
  local b="$1"
  local ymd="${b:10:10}"                         # "YYYY-MM-DD"
  local dow start                                # 0=Sun, 1=Mon, ...
  dow=$(date -u -d "$ymd" +%w)
  start=$(date -u -d "$ymd - ${dow} days" +%Y-%m-%d)
  echo "$start"
}

# --- 対象ファイルの一覧（新しい順に整列） --------------------------------------
# find の -printf は環境差が出るため使わず、シェルのグロブで列挙
# 行フォーマット: "epoch<TAB>basename<TAB>fullpath"
shopt -s nullglob
mapfile -t L < <(
  for f in "$DIR"/sqlbackup-*.sql; do
    base=$(basename "$f")
    ep=$(parse_epoch_from_basename "$base")
    [[ -n "$ep" ]] && printf "%s\t%s\t%s\n" "$ep" "$base" "$f"
  done | sort -r -n -k1,1
)
# 対象がなければ終了（正常）
[[ ${#L[@]} -eq 0 ]] && { echo "no target files"; exit 0; }

# --- 判定用の連想配列 -----------------------------------------------------------
# KEEP: ファイルパス → 1
# REASON: ファイルパス → hourly/daily/weekly/monthly/yearly
# SEEN_*: 各期間キー → 既に採用済みフラグ
declare -A KEEP REASON SEEN_H SEEN_D SEEN_W SEEN_M SEEN_Y

# --- GFS 判定（新しいものから順に見る → 各期間で最初の1件を採用） -----------
for line in "${L[@]}"; do
  IFS=$'\t' read -r ep base full <<<"$line"
  # すでに KEEP 済みならスキップ
  [[ -n "${KEEP[$full]:-}" ]] && continue

  if (( ep >= cut_hour )); then
    # Hourly 範囲: 同じ「YYYY-MM-DDTHH」が未採用なら KEEP
    k=$(hour_key_from_base "$base")
    [[ -z "${SEEN_H[$k]:-}" ]] && { KEEP["$full"]=1; REASON["$full"]="hourly";  SEEN_H["$k"]=1; }
  elif (( ep >= cut_day )); then
    # Daily 範囲: 同じ「YYYY-MM-DD」が未採用なら KEEP
    k=$(day_key_from_base "$base")
    [[ -z "${SEEN_D[$k]:-}" ]] && { KEEP["$full"]=1; REASON["$full"]="daily";   SEEN_D["$k"]=1; }
  elif (( ep >= cut_week )); then
    # Weekly 範囲: 同じ「週頭(日曜)YYYY-MM-DD」が未採用なら KEEP
    k=$(week_key_from_base "$base")
    [[ -z "${SEEN_W[$k]:-}" ]] && { KEEP["$full"]=1; REASON["$full"]="weekly";  SEEN_W["$k"]=1; }
  elif (( ep >= cut_month )); then
    # Monthly 範囲: 同じ「YYYY-MM」が未採用なら KEEP
    k=$(month_key_from_base "$base")
    [[ -z "${SEEN_M[$k]:-}" ]] && { KEEP["$full"]=1; REASON["$full"]="monthly"; SEEN_M["$k"]=1; }
  elif (( ep >= cut_year )); then
    # Yearly 範囲: 同じ「YYYY」が未採用なら KEEP
    k=$(year_key_from_base "$base")
    [[ -z "${SEEN_Y[$k]:-}" ]] && { KEEP["$full"]=1; REASON["$full"]="yearly";  SEEN_Y["$k"]=1; }
  fi
done

# --- 出力＆削除 ---------------------------------------------------------------
# ここまでで KEEP に含まれないものは DEL 対象
echo "ACT   TIMESTAMP(UTC)         FILE"
keep=0; del=0
for line in "${L[@]}"; do
  IFS=$'\t' read -r ep base full <<<"$line"
  ts=$(fmt_ts_utc "$ep")
  if [[ -n "${KEEP[$full]:-}" ]]; then
    # KEEP: 理由（hourly/daily/...）も表示
    printf "KEEP  %-20s  %s (%s)\n" "$ts" "$base" "${REASON[$full]}"
    ((++keep))  # set -e 下でも安全（前置インクリメント）
  else
    # DEL: ドライラン時は削除しない。--apply のときのみ削除する
    printf "DEL   %-20s  %s\n" "$ts" "$base"
    ((++del))
    [[ "$APPLY" == "yes" ]] && rm -f -- "$full"
  fi
done

# ドライラン表示
[[ "$APPLY" == "no" ]] && echo "(dry-run)"

# サマリ
echo "SUMMARY keep=$keep del=$del"
