#date: 2025-05-06T17:02:57Z
#url: https://api.github.com/gists/57435bcdcb4358927e35dffd9faae706
#owner: https://api.github.com/users/VladislavSmolyanoy

#!/usr/bin/env bash
# zip_benchmark.sh  —  v2025‑05‑06  (quiet‑safe)
#
# Same purpose as previous version but now detects whether `zipcloak`
# supports `-q` (quiet). On macOSʼs default Info‑ZIP 3.0 the flag does
# **not** exist, so we omit it to avoid exit‑16 errors.
#
#  ‣ Speed ranking
#  ‣ Size  ranking
#  ‣ Efficiency (MiB × s)
#
# Usage:
#   ./zip_benchmark.sh <SOURCE> [PASSWORD|-] [THREADS]
#
# ----------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: "**********"
  exit 1
fi

SRC="$1"
PASS="${2:-}"
THREADS="${3:-0}"
WITHOUT_PW=false
[[ -z "$PASS" || "$PASS" == "-" ]] && WITHOUT_PW=true

[[ -d "$SRC" || -f "$SRC" ]] || { echo "Source not found: $SRC" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# Detect if zipcloak supports -q
ZIPCLOAK_Q=""
if have zipcloak; then
  if zipcloak -h 2>&1 | grep -q -- ' -q'; then
    ZIPCLOAK_Q="-q"
  fi
fi

RESULTS=()   # label seconds bytes

measure() {
  local label="$1"; shift
  local outfile="$1"; shift

  rm -f "$outfile"
  echo "▶ $label ..."
  local tmp_out
  tmp_out=$(mktemp)
  set +e
  /usr/bin/time -p "$@" 2> >(tee "$tmp_out")
  local exit_code=$?
  set -e
  local secs=""
  if grep -q '^real ' "$tmp_out"; then
    secs=$(awk '/^real /{print $2}' "$tmp_out")
  fi
  rm -f "$tmp_out"

  if [[ $exit_code -ne 0 ]]; then
    echo "   ⚠️  ${label} failed (exit $exit_code) – skipped"
  else
    local bytes
    bytes=$(wc -c < "$outfile")
    RESULTS+=("$label $secs $bytes")
    printf "   ↳ %s finished in %ss, size %.2f MiB\n\n" \
           "$label" "$secs" "$(echo "$bytes / 1048576" | bc -l)"
  fi
}

###########################################################################
# 1. zip
###########################################################################
if have zip; then
  for lvl in 1 6 9; do
    if $WITHOUT_PW; then
      measure "zip-$lvl" "zip_l${lvl}.zip" \
        zip -qr -${lvl} "zip_l${lvl}.zip" "$SRC"
    else
      measure "zip-$lvl" "zip_l${lvl}.zip" \
        zip -qr -${lvl} -P "$PASS" "zip_l${lvl}.zip" "$SRC"
    fi
  done
else
  echo "zip not found – skipped"
fi

###########################################################################
# 2. p7zip
###########################################################################
if have 7z; then
  mtflag=""
  [[ "$THREADS" != "0" ]] && mtflag="-mmt=$THREADS"
  for lvl in 1 5 9; do
    if $WITHOUT_PW; then
      measure "p7zip-$lvl" "p7zip_l${lvl}.zip" \
        7z a -tzip "p7zip_l${lvl}.zip" "$SRC" $mtflag -mx=${lvl}
    else
      measure "p7zip-$lvl" "p7zip_l${lvl}.zip" \
        7z a -tzip "p7zip_l${lvl}.zip" "$SRC" -p"$PASS" -mem=AES256 $mtflag -mx=${lvl}
    fi
  done
else
  echo "p7zip not found – skipped"
fi

###########################################################################
# 3. pigz + zip(/zipcloak)
###########################################################################
if have pigz && have zip && have gzip; then
  pigz_threads=""
  [[ "$THREADS" != "0" ]] && pigz_threads="-p $THREADS"
  for lvl in 1 6 9; do
    if $WITHOUT_PW; then
      measure "pigz-$lvl" "pigz_l${lvl}.zip" bash -c '
        tar cf - "'"$SRC"'" | pigz '"$pigz_threads"' -'"$lvl"' > tmp_pigz.tar.gz && \
        gzip -dc tmp_pigz.tar.gz | zip -q -j -0 pigz_l'"${lvl}"'.zip - && \
        rm tmp_pigz.tar.gz'
    else
      have zipcloak || { echo "zipcloak not found – skipping pigz+zipcloak"; continue; }
      measure "pigz+zipcloak-$lvl" "pigz_l${lvl}.zip" bash -c '
        tar cf - "'"$SRC"'" | pigz '"$pigz_threads"' -'"$lvl"' > tmp_pigz.tar.gz && \
        gzip -dc tmp_pigz.tar.gz | zip -q -j -0 pigz_l'"${lvl}"'.zip - && \
        zipcloak '"$ZIPCLOAK_Q"' -P "'"$PASS"'" pigz_l'"${lvl}"'.zip >/dev/null && \
        rm tmp_pigz.tar.gz'
    fi
  done
else
  echo "pigz and/or zip not found – skipped"
fi

###########################################################################
# 4. zstd + (optional) gpg
###########################################################################
if have zstd; then
  zstd_threads="-T$THREADS"
  [[ "$THREADS" == "0" ]] && zstd_threads="-T0"
  for tag in 1 3 19; do
    case $tag in
      1) zlvl="-1" ;; 3) zlvl="-3" ;; 19) zlvl="-19" ;;
    esac
    if $WITHOUT_PW; then
      measure "zstd-$tag" "zstd_l${tag}.tar.zst" bash -c '
        tar cf - "'"$SRC"'" | zstd '"$zstd_threads"' '"$zlvl"' -q -o zstd_l'"${tag}"'.tar.zst'
    else
      have gpg || { echo "gpg not found – skipping zstd+gpg"; continue; }
      measure "zstd+gpg-$tag" "zstd_l${tag}.tar.zst.gpg" bash -c '
        tar cf - "'"$SRC"'" | zstd '"$zstd_threads"' '"$zlvl"' -q | \
        gpg --batch --yes --passphrase "'"$PASS"'" --symmetric \
            --cipher-algo AES256 -o zstd_l'"${tag}"'.tar.zst.gpg'
    fi
  done
else
  echo "zstd not found – skipped"
fi

###########################################################################
# Summaries
###########################################################################
if [[ ${#RESULTS[@]} -eq 0 ]]; then
  echo "No successful runs."
  exit 0
fi

report_file=$(mktemp)
for entry in "${RESULTS[@]}"; do
  echo "$entry" >> "$report_file"
done

echo
echo "====================  SPEED (s)  ===================="
printf "%-25s %10s\n" "METHOD" "SECONDS"
sort -k2 -n "$report_file" | awk '{printf "%-25s %10s\n",$1,$2}'

echo
echo "====================  SIZE (MiB) ===================="
printf "%-25s %10s\n" "METHOD" "MiB"
sort -k3 -n "$report_file" | awk '{printf "%-25s %10.2f\n",$1,$3/1048576}'

echo
echo "============  EFFICIENCY  (MiB × s)  ==============="
printf "%-25s %10s\n" "METHOD" "MiB*s"
awk '{eff=$2*($3/1048576); printf "%-25s %10.2f\n",$1,eff}' "$report_file" | sort -k2 -n

rm -f "$report_file"k2 -n

rm -f "$report_file"