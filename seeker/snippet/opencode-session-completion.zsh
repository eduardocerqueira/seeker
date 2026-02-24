#date: 2026-02-24T17:31:59Z
#url: https://api.github.com/gists/59c89e8eaf4d9d80c132f226077441f2
#owner: https://api.github.com/users/audionerd

# requires fzf and jq
# fzf-powered completion for: opencode -s / --session
_opencode_session_ids_fzf() {
  local selected id
  selected="$(
    opencode session list --format json 2>/dev/null \
      | jq -r '
          def fmt($ms):
            ($ms / 1000 | floor | strflocaltime("%-I:%M %p · %-m/%-d/%Y"));
          .[]
          | select(.id | startswith("ses_"))
          | "\(.id)\t\(.title // "")\t\(fmt(.updated))"
        ' \
      | awk -F'\t' '{
          title=$2
          if (length(title) > 70) title=substr(title,1,67) "..."
          printf "%-34s  %-70s  %22s\n", $1, title, $3
        }' \
      | fzf \
          --height=40% \
          --layout=reverse \
          --border=none \
          --prompt='> ' \
          --pointer='▌' \
          --marker=''
  )" || return 1
  id="${selected%% *}"
  [[ "$id" == ses_* ]] && compadd -- "$id"
}
# Wrap default opencode completion so only -s/--session uses fzf picker
_opencode_completion_with_session_picker() {
  local curcontext="$curcontext" state line
  typeset -A opt_args
  _arguments -C \
    '(-s --session)'{-s,--session}'[session id]:session:_opencode_session_ids_fzf' \
    '*::arg:_opencode_yargs_completions'
}
compdef _opencode_completion_with_session_picker opencode
