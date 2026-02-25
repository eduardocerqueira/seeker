#date: 2026-02-25T17:40:46Z
#url: https://api.github.com/gists/947c4b97754f30b854772aaf79889c0b
#owner: https://api.github.com/users/Ge0rg3

#!/usr/bin/env bash
# conda.sh - interactive conda env selector
# Must be sourced:  source /path/to/conda.sh

# Must be sourced for activation to affect the current shell.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced: source conda.sh" >&2
  exit 1
fi

# Selection runs in a subshell so strict-mode does NOT leak into your shell
_selected="$(
  (
    set -euo pipefail

    fail() { echo "Error: $*" >&2; exit 1; }

    ensure_conda() {
      command -v conda >/dev/null 2>&1 || fail "conda not found in PATH. Run: conda init bash, then restart your shell."
    }

    get_pyver() {
      local prefix="$1"
      local pbin="$prefix/bin/python"
      if [[ -x "$pbin" ]]; then
        "$pbin" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null && return 0
      fi
      conda run --prefix "$prefix" python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || echo "N/A"
    }

    collect_envs() {
      mapfile -t ENVS < <(
        conda env list 2>/dev/null \
          | awk '
              NF==0 {next}
              $1 ~ /^#/ {next}
              {
                name=$1
                if ($2=="*") { path=$3 } else { path=$2 }
                if (name != "" && path != "") print name "\t" path
              }
            '
      )
      ((${#ENVS[@]} > 0)) || fail "No conda environments found."
    }

    build_display_lines() {
      LINES=()
      for row in "${ENVS[@]}"; do
        name="${row%%$'\t'*}"
        path="${row#*$'\t'}"
        pyver="$(get_pyver "$path")"
        LINES+=("${name}"$'\t'"py${pyver}"$'\t'"${path}")
      done
    }

    choose_with_fzf() {
      printf "%s\n" "${LINES[@]}" \
        | fzf --height=40% --layout=reverse --border \
              --prompt="conda env> " \
              --header=$'NAME\tPYTHON\tPATH' \
              --delimiter=$'\t' \
              --with-nth=1,2,3
    }

    choose_with_select() {
      echo "fzf not found; using numbered menu." >&2
      echo >&2
      local i=1
      for l in "${LINES[@]}"; do
        local name py
        name="${l%%$'\t'*}"
        py="$(awk -F'\t' '{print $2}' <<<"$l")"
        printf "%2d) %s (%s)\n" "$i" "$name" "$py" >&2
        ((i++))
      done
      echo >&2
      read -rp "Select env number: " n
      [[ "$n" =~ ^[0-9]+$ ]] || fail "Invalid selection."
      (( n >= 1 && n <= ${#LINES[@]} )) || fail "Selection out of range."
      printf "%s\n" "${LINES[$((n-1))]}"
    }

    select_env() {
      if command -v fzf >/dev/null 2>&1; then
        choose_with_fzf || true
      else
        choose_with_select || true
      fi
    }

    ensure_conda
    collect_envs
    build_display_lines

    selected="$(select_env)"
    [[ -n "${selected}" ]] || exit 0
    printf "%s\n" "$selected"
  )
)" || return 0

# user cancelled
[[ -n "${_selected}" ]] || return 0

sel_path="$(awk -F'\t' '{print $3}' <<<"$_selected")"
sel_name="$(awk -F'\t' '{print $1}' <<<"$_selected")"
sel_py="$(awk -F'\t' '{print $2}' <<<"$_selected")"

conda activate "$sel_path"
printf "Activated: %s (%s)\n" "$sel_name" "$sel_py"
