#date: 2025-11-21T17:02:23Z
#url: https://api.github.com/gists/7fe11a156ac483a83f083112bf54fcb8
#owner: https://api.github.com/users/johnlindquist

# =========================
# John’s Zsh Profile (clean)
# Last updated: 2025-10-09
# =========================

#### 0) Alias / shorthands used very early
# alias ca="cursor-agent"

#### 1) Core environment & PATH (de‑duplicated, order preserved)
# Use zsh's unique path array to avoid duplicates automatically.
typeset -U path

# Common homes
export ZSH="$HOME/.oh-my-zsh"
export PNPM_HOME="$HOME/Library/pnpm"
export BUN_INSTALL="$HOME/.bun"

# Build PATH with highest-priority tools first, then append existing $path.
path=(
  # Node/pnpm/bun (highest priority to avoid symlink issues)
  "$PNPM_HOME"
  "$HOME/Library/pnpm/nodejs/23.6.1/bin"
  "$BUN_INSTALL/bin"

  /opt/homebrew/bin
  /usr/local/bin
  /usr/bin
  /bin

  # CLIs / Dev tools
  "$HOME/.codeium/windsurf/bin"
  "$HOME/.lmstudio/bin"
  "$HOME/.pub-cache/bin"
  "$HOME/.npm-global/bin"
  "$HOME/.local/bin"
  "$HOME/dev/agents/bin"
  "$HOME/dev/claude-workshop-live/bin"
  "$HOME/dev/claude-workshop-live/jq-filters"
  "$HOME/dev/pack/bin"

  $path  # keep anything the OS or other tools already added
)
export PATH="${(j/:/)path}"

# Keep PNPM at high precedence
case ":$PATH:" in
  *":$PNPM_HOME:"*) ;;
  *) export PATH="$PNPM_HOME:$PATH" ;;
esac

# Warnings / noise
export PYTHONWARNINGS=ignore::DeprecationWarning

#### 2) Oh My Zsh
ZSH_THEME="robbyrussell"
plugins=(git)
source "$ZSH/oh-my-zsh.sh"

# Completions (user)
fpath+=~/.zfunc
autoload -Uz compinit; compinit

#### 3) Quality-of-life aliases
# alias cursor="/usr/local/bin/cursor"
alias w="$HOME/.codeium/windsurf/bin/windsurf"
alias ww="$HOME/.codeium/windsurf/bin/windsurf ~/dev/windsurf"

alias z="zed"
alias zz="zed ~/.zshrc"
alias zk="zed ~/.config/karabiner.edn"
alias k="/usr/local/bin/cursor ~/.config/karabiner.edn"
alias s="source ~/.zshrc"
alias pup="pnpm dlx npm-check-updates -i -p pnpm"

# Karabiner profile switching
alias kar1="'/Library/Application Support/org.pqrs/Karabiner-Elements/bin/karabiner_cli' --select-profile 'Default'"
alias kar2="'/Library/Application Support/org.pqrs/Karabiner-Elements/bin/karabiner_cli' --select-profile 'Default profile'"

alias pke="pkill Electron"                # quick kill Electron apps
alias nx='pnpm dlx nx'                    # Nx wrapper
alias tk='tail -f ~/Library/Logs/ScriptKit/$1.log'
alias config='/usr/bin/git --git-dir=$HOME/.config/.git --work-tree=$HOME/.work' # dotfiles
alias opus='ENABLE_BACKGROUND_TASKS=1 claude --model opus'

#### 4) Shell history / enhancements
. "$HOME/.atuin/bin/env"
eval "$(atuin init zsh)"

#### 5) 1Password plugins (op)
source $HOME/.config/op/plugins.sh

#### 6) Editor/workflow helpers
unalias c 2>/dev/null
c() {
  if [ $# -eq 0 ]; then
    /usr/local/bin/cursor .
  else
    if [ ! -e "$1" ]; then
      if [[ "$1" == .* || "$1" == *.* ]]; then
        touch "$1"
      else
        mkdir -p "$1"
      fi
    fi
    /usr/local/bin/cursor "$@"
  fi
}

ca(){
  env -u CURSOR_CLI -u CURSOR_AGENT $HOME/.local/bin/cursor-agent "$@"
}

#### 7) Conventional commit helpers
unalias fix feat chore push 2>/dev/null
cfix()   { local scope="$1" message="$2"; git add . && git commit -m "fix($scope): $message"; }
fix()    { local scope="$1" message="$2"; git add . && git commit -m "fix($scope): $message" && git push; }
feat()   { local scope="$1" message="$2"; git add . && git commit -m "feat($scope): $message" && git push; }
chore()  { local scope="$1" message="$2"; git add . && git commit -m "chore($scope): $message" && git push; }
push()   { git add . && git commit -m "fix: tweak" && git push; }

#### 8) Cursor + Windsurf helpers
takeAndWindsurf() { take "$1" && windsurf "$1"; }

clone(){
  local repo="$1"
  local dir="${2:-${repo##*/}}"
  gh repo clone "https://github.com/$repo" "$dir"
  w "$dir"
  cd "$dir" || return
  pnpm i
}

kdev(){
  cd ~/dev/kit || return
  pnpm build
  cd - || return
  pnpm dev
}

share-react-project() {
  if [[ -z "$1" ]]; then
    echo "Usage: share-react-project <project_name>"; return 1
  fi
  local project_name="$1"
  local github_username
  github_username=$(gh api /user --jq '.login')

  echo "Creating Vite project: $project_name"
  pnpm create vite "$project_name" --template react
  cd "$project_name" || return

  git init
  git add .
  git commit -m "Initial commit"

  local codesandbox_link="https://codesandbox.io/p/github/${github_username}/${project_name}"
  {
    echo ""
    echo "## CodeSandbox"
    echo "[![Open in CodeSandbox](https://assets.codesandbox.io/github/button-edit-blue.svg)](${codesandbox_link})"
  } >> README.md

  git add README.md
  git commit -m "Add CodeSandbox link"

  gh repo create "$github_username/$project_name" --public
  git push -u origin main

  echo "Project '$project_name' created!"
  echo "GitHub: https://github.com/$github_username/$project_name"
  echo "CodeSandbox: $codesandbox_link"
}

pinit23(){
  pnpm init
  pnpm pkg set type=module
  pnpm pkg set scripts.dev="node --env-file=.env --no-warnings index.ts"
  pnpm set --location project use-node-version 23.6.1
  pnpm add -D @types/node @tsconfig/node23 @tsconfig/strictest
  pnpm add dotenv
  echo 'TEST_API_KEY=Successfully loaded .env' > .env
  pnpm dlx gitignore Node
  cat > tsconfig.json <<'JSON'
{
  "$schema": "https://json.schemastore.org/tsconfig",
  "extends": ["@tsconfig/node23/tsconfig.json", "@tsconfig/strictest/tsconfig.json"]
}
JSON
  cat > index.ts <<'TS'
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      TEST_API_KEY: string;
    }
  }
}
console.log(`${process.env.TEST_API_KEY || "Failed to load .env"}`);
TS
  mkdir -p logs
  pnpm dev
  git init && git add . && git commit -m "(feat):project setup"
}

#### 9) GitHub code search (with logs + snippets)
ghsearch() {
  local ORIGINAL_PATH="$PATH"
  local debug=1
  local timestamp=$(/bin/date +%Y%m%d-%H%M%S)
  local log_dir="$HOME/searches/logs"
  local log_file="$log_dir/ghsearch-$timestamp.log"
  /bin/mkdir -p "$log_dir" 2>/dev/null

  log() { local level="$1" msg="$2"; [[ "$level" == DEBUG && $debug -eq 0 ]] && return; echo "[$level] $msg" | /usr/bin/tee -a "$log_file"; }

  log DEBUG "Starting ghsearch"
  log DEBUG "Command: ghsearch $*"

  export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$ORIGINAL_PATH"
  log DEBUG "PATH: $PATH"

  local query="$*"
  [[ -z "$query" ]] && { log ERROR "No query provided"; return 1; }
  [[ "$query" =~ [^a-zA-Z0-9[:space:]/_.-] ]] && log WARN "Query contains special chars: $query"

  local sanitized_query
  sanitized_query=$(/bin/echo "$query" | /usr/bin/tr -c '[:alnum:]_-' '_' | /usr/bin/sed 's/_*$//')
  local results_dir="$HOME/searches"
  local results_file="$results_dir/$sanitized_query-$timestamp.md"
  /bin/mkdir -p "$results_dir" 2>/dev/null
  log INFO "Saving to: $results_file"

  local gh_path=$(/usr/bin/which gh 2>/dev/null)
  local jq_path=$(/usr/bin/which jq 2>/dev/null)
  local curl_path=$(/usr/bin/which curl 2>/dev/null)
  [[ -z $gh_path || -z $jq_path || -z $curl_path ]] && { log ERROR "Install gh, jq, curl"; return 1; }

  log INFO "Executing GitHub search"
  local search_output
  search_output=$(/opt/homebrew/bin/gh search code "$query" --json path,repository,url --limit 30)
  local gh_exit="$?"
  [[ "$gh_exit" -ne 0 ]] && { log ERROR "gh failed ($gh_exit)"; log ERROR "Raw: $search_output"; return 1; }

  if ! /bin/echo "$search_output" | /opt/homebrew/bin/jq . >/dev/null 2>&1; then
    log ERROR "Invalid JSON"; log ERROR "Raw: $search_output"; return 1
  fi

  local count
  count=$(/bin/echo "$search_output" | /opt/homebrew/bin/jq 'length')
  log DEBUG "Found $count results"

  {
    /bin/echo "# GitHub Code Search Results"
    /bin/echo "Query: \`$query\`"
    /bin/echo "Date: $(/bin/date)"
    /bin/echo
    if [ "$count" -eq 0 ]; then
      /bin/echo "No results found."
    else
      /bin/echo "Found $count results. Showing snippets."
      /bin/echo
      /bin/echo "## Results"
      /bin/echo
      /bin/echo "$search_output" \
      | /opt/homebrew/bin/jq -r \
        '.[] | "### [\(.repository.nameWithOwner)](\(.repository.url))\n\nFile: [\(.path)](\(.url))\n\n```" + (.path | match("\\.[a-zA-Z0-9]+$") | .string[1:] // "") + "\n# Content from \(.path):\n" + (.url | sub("github.com"; "raw.githubusercontent.com") | sub("/blob/"; "/")) + "\n"' \
      | while read -r line; do
          if [[ "$line" =~ ^https ]]; then
            content=$(/usr/bin/curl -s -L "$line")
            if [ -n "$content" ]; then
              /bin/echo "$content" | /usr/bin/awk '{printf "%4d: %s\n", NR, $0}' | /usr/bin/head -n 50
              if [ "$(/bin/echo "$content" | /usr/bin/wc -l)" -gt 50 ]; then
                /bin/echo "... (truncated)"
              fi
            else
              /bin/echo "Failed to fetch $line"
            fi
            /bin/echo '```'
            /bin/echo
            /bin/echo "---"
            /bin/echo
          else
            /bin/echo "$line"
          fi
        done
    fi
  } > "$results_file"

  log DEBUG "Opening results in Cursor"
  if [ -f "$results_file" ]; then
    if ! /Applications/Cursor.app/Contents/MacOS/Cursor "$results_file" 2>/dev/null; then
      log ERROR "Open failed. Use: cursor '$results_file'"
    fi
  fi

  export PATH="$ORIGINAL_PATH"
  log DEBUG "ghsearch complete"
}

#### 10) Git worktree helpers
wtree() {
  local install_deps=false
  local branches=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -p|--pnpm) install_deps=true; shift ;;
      *) branches+=("$1"); shift ;;
    esac
  done
  [[ ${#branches[@]} -eq 0 ]] && { echo "Usage: wtree [-p] branch1 [branch2...]"; return 1; }

  local current_branch repo_root repo_name worktree_parent="$HOME/dev"
  current_branch=$(git rev-parse --abbrev-ref HEAD) || { echo "Not a git repo"; return 1; }
  repo_root=$(git rev-parse --show-toplevel) || { echo "Cannot find repo root"; return 1; }
  repo_name=$(basename "$repo_root")
  mkdir -p "$worktree_parent" || { echo "Cannot create $worktree_parent"; return 1; }

  for branch in "${branches[@]}"; do
    local target="$worktree_parent/${repo_name}-${branch}"
    echo "Processing: $branch → $target"
    if git worktree list | grep -q "^${target}[[:space:]]"; then
      echo "Worktree exists at ${target}. Skipping '${branch}'."
      continue
    fi
    git show-ref --verify --quiet "refs/heads/${branch}" || git branch "${branch}" || { echo "Failed creating '${branch}'"; continue; }
    git worktree add "$target" "${branch}" || { echo "Failed add worktree '${branch}'"; continue; }
    if $install_deps; then
      echo "Installing deps in ${target}..."
      (cd "$target" && pnpm install) || echo "Warning: pnpm install failed"
    fi
    if type cursor >/dev/null 2>&1; then cursor "$target"; else echo "Worktree: ${target}"; fi
    echo "-----------------------------------------------------"
  done
}

wtmerge() {
  [[ $# -eq 1 ]] || { echo "Usage: wtmerge <branch-to-keep>"; return 1; }
  local keep="$1" repo_root repo_name worktree_parent="$HOME/dev"
  repo_root=$(git rev-parse --show-toplevel) || { echo "Not a git repo"; return 1; }
  repo_name=$(basename "$repo_root")

  local worktrees=()
  while IFS= read -r line; do
    local wt_path; wt_path=$(echo "$line" | awk '{print $1}')
    [[ "$wt_path" == "$worktree_parent/${repo_name}-"* ]] && worktrees+=("$wt_path")
  done < <(git worktree list)

  local target=""
  for wt in "${worktrees[@]}"; do
    [[ "$wt" == "$worktree_parent/${repo_name}-${keep}" ]] && target="$wt" && break
  done
  [[ -z "$target" ]] && { echo "No worktree for '${keep}' under ${worktree_parent}"; return 1; }

  echo "Checking uncommitted changes in '${keep}'..."
  if ! ( cd "$target" && git diff --quiet && git diff --cached --quiet ); then
    ( cd "$target" && git add . && git commit -m "chore: auto-commit '${keep}' before merge" ) || { echo "Auto-commit failed"; return 1; }
  fi

  echo "Switching to main and merging '${keep}'..."
  git checkout main || { echo "Failed to checkout main"; return 1; }
  git merge "${keep}" -m "feat: merge '${keep}'" || { echo "Merge failed"; return 1; }

  echo "Cleaning worktrees..."
  for wt in "${worktrees[@]}"; do
    local wt_branch; wt_branch=$(basename "$wt"); wt_branch=${wt_branch#${repo_name}-}
    git worktree remove "$wt" --force || echo "Warning: couldn't remove $wt"
    [[ "$wt_branch" != "main" ]] && git branch -D "$wt_branch" || true
  done
  echo "Done."
}

#### 11) Small git helpers
spike(){
  local base; base=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || return 1
  local branch; branch="${1:-spike/${base}-$(date +%s)}"
  echo "🌵 Spiking to $branch ..."
  git switch -c "$branch" || return 1
  git add -A || return 1
  git commit -m "spike(${base}): ${branch##*/}" || return 1
  git switch "$base"
}

#### 12) AI/search helpers
google() {
  [[ $# -eq 0 ]] && { echo "Usage: google \"search query\""; return 1; }
  local search_query="$*"
  local tmpdir; tmpdir=$(mktemp -d)
  local pids=() i=0
  local searches=("$search_query")

  for search in "${searches[@]}"; do
    [[ -z "$search" ]] && continue
    echo "Googling... $search"
    (
      claude -p "web_search for <query>$search</query> and summarize the results" --allowedTools="web_search" > "$tmpdir/result_$i.txt"
    ) &
    pids+=($!)
    ((i++))
  done

  for pid in "${pids[@]}"; do wait "$pid"; done

  local results=""
  for file in "$tmpdir"/result_*.txt; do results+=$(cat "$file"); results+=$'\n'; done
  local final_report
  final_report=$(claude -p "Write a blog post based on these results for <query>$search_query</query>: $results")
  echo "$final_report"
}

claude_chain(){
  local -a c=(claude --permission-mode acceptEdits -p)
  local m='then git commit'
  local p="$*"
  "${c[@]}" "$p, $m" && "${c[@]}" "Review and improve latest commit based on '$p', $m"
}

claudepool(){
  claude --append-system-prompt "Talk like a caffeinated Deadpool with sadistic commentary and comically PG-13 rated todo lists."
}

organize(){
  local system_prompt=$(cat <<'EOF'
You are an expert obsidian project organizer. Follow:
1) Shallow folders; organize via links & tags
2) Clear descriptive titles; remove dates
3) Promote clusters to MOC hubs; review weekly
Use 5+ subagents. Commit frequently.
EOF
)
  ENABLE_BACKGROUND_TASKS=1 claude --model opus --dangerously-skip-permissions --append-system-prompt "$system_prompt" -p "Organize this project." --output-format stream-json --verbose
}

curate(){
  local system_prompt=$(cat <<'EOF'
You are an expert obsidian curator. Focus on Knowledge Paths/Hubs.
CRITICAL: Condense duplicates. Sync names/links with rg.
Steps: tree → rg → curate → commit/push. Use 3+ subagents.
EOF
)
  ENABLE_BACKGROUND_TASKS=1 claude --model opus --dangerously-skip-permissions --append-system-prompt "$system_prompt" -p "Curate this project." --output-format stream-json --verbose
}

verify(){
  local system_prompt=$(cat <<'EOF'
You verify files against the web.
Steps: tree → 3 random files → verify → fix/create → commit.
Add frontmatter "last-verified" and "## Verifications" with sources.
EOF
)
  ENABLE_BACKGROUND_TASKS=1 claude --model opus --dangerously-skip-permissions --append-system-prompt "$system_prompt" -p "Verify this project." --output-format stream-json --verbose
}

# --- Preferred (new) research function (keeps your override intent) ---
research(){
  [[ $# -eq 0 ]] && { echo "Usage: research <topic1> [topic2] ..."; return 1; }
  local project_name="$(basename "$PWD")"
  for topic in "$@"; do
    echo "🔎 Researching '$topic' for '$project_name'..."
    local content
    content="$(claude --model opus --dangerously-skip-permissions -p "Expert technical researcher. Obsidian-ready markdown for '${topic}' aimed at TypeScript devs. Include headings, bullets, code, references.")"
    echo "$content" | basic-memory --project "$project_name" tool write-note --title "$topic" --folder "research"
  done
  basic-memory --project "$project_name" tool recent-activity --timeframe 1d
}

# --- Quartz research helpers (container-use) ---
quartz_research() {
  local MCP_CONFIG=$(cat <<'EOF'
{
  "mcpServers": {
    "container-use": { "type": "stdio", "command": "container-use", "args": ["stdio"] }
  }
}
EOF
)
  [[ $# -eq 0 ]] && { echo "Usage: quartz_research <topic1> [topic2] ..."; return 1; }
  local project_name="$(basename "$PWD")"
  local quartz_prompt=$(cat <<'EOF'
TOOLS:
ALWAYS use ONLY Environments for ANY file/code/shell ops—NO EXCEPTIONS.
Do NOT install/use git cli via environment_run_cmd; use env tools.
You MUST tell the user to use: container-use log <env_id> and container-use checkout <env_id>.

OBJECTIVE:
For each topic, create/update Quartz Markdown notes for expert devs in ≤5 min reading.

HARD CONSTRAINTS:
1) Atomic commits (git add <file> && git commit -m "<slug>: <summary>")
2) "## Sources" list with full URLs
3) ≤400 lines per note
4) De-duplicate via grep before writing
5) Exactly three sub-agents: researcher, summarizer, note-writer
6) Stop when coverage ≥0.8 or 5 successive commits have <10 LOC

OUTPUT:
Return concise per-commit status lines:
[OK] <file_path> (<LOC_delta>) – <summary>
EOF
)
  for topic in "$@"; do
    echo "🧠 quartz_research → '$topic' in '$project_name'"
    ENABLE_BACKGROUND_TASKS=1 cl "$topic" \
      --model opus \
      --append-system-prompt "$quartz_prompt" \
      --mcp-config "$MCP_CONFIG" \
      --print | claude --print --append-system-prompt "Merge using git checkout main then container-use merge <branch>"
  done
}

basic_memory_consistency() {
  local project_name="$(basename "$PWD")"
  { bm project add "$project_name" . 2>&1 || true; } | grep -qi "already exists" && echo "Already exists, continuing..."
  local bm_prompt=$(cat <<'EOF'
SYSTEM:
You are Basic‑Memory‑Agent v2 operating in "$project_name".
OBJECTIVE: Ensure consistency, organization, metadata across notes.

CONSTRAINTS:
1) Atomic commits
2) Use MCP tools (basic-memory) only
3) ≤400 lines per file
4) Shallow hierarchy: notes/, docs/, research/
5) Stop when clean scan has no inconsistencies

OUTPUT:
[OK] <action> <file_path> – <summary>
EOF
)
  ENABLE_BACKGROUND_TASKS=1 claude \
    --model opus \
    --dangerously-skip-permissions \
    --allowedTools="run_terminal_cmd" \
    --append-system-prompt "$bm_prompt" \
    -p "$topic" \
    --output-format stream-json \
    --verbose \
    --mcp-config "{\"mcpServers\":{\"basic-memory\":{\"command\":\"bm\",\"args\":[\"--project\",\"$project_name\",\"mcp\"]}}}"
}

# Cyclic “auto” Quartz (kept intact)
auto_quartz() {
  set -euo pipefail
  local base_sleep=300 failure_penalty=1800 max_retries=5 retry_count=0
  run_and_sleep() {
    local cmd_name="$1"; shift
    local extra_sleep=0
    local output_file; output_file=$(mktemp)
    if "$@" 2>&1 | tee "$output_file"; then
      if grep -q "EXIT_FAILURE" "$output_file"; then
        echo "⚠️  $cmd_name reported EXIT_FAILURE"; extra_sleep=$failure_penalty; ((retry_count++))
      else retry_count=0; fi
    else echo "⚠️  $cmd_name failed"; extra_sleep=$failure_penalty; ((retry_count++)); fi
    rm -f "$output_file"
    [[ $retry_count -ge $max_retries ]] && { echo "❌ Max retries reached."; return 1; }
    local total_sleep=$((base_sleep + extra_sleep))
    echo "😴 Sleeping $((total_sleep / 60)) min... (failures: $retry_count/$max_retries)"
    sleep $total_sleep
  }
  for i in {1..100}; do
    echo "🔄 Run #$i (retries: $retry_count/$max_retries)"
    local instructions_file="$(date +%Y%m%d_%H%M%S)-instructions.txt"
    local gemini_prompt=$(cat <<'EOF'
SYSTEM: Gemini‑Tasks‑Bot v1. Output EXACTLY a JSON array of 3 task objects—no prose.
... (schema and rules omitted for brevity in this profile)
EOF
)
    local random_files file_digests=""
    if [ -d "./content" ]; then
      mapfile -t random_files_array < <(find ./content -name "*.md" -type f | shuf -n 3)
      if [ ${#random_files_array[@]} -gt 0 ]; then
        for file in "${random_files_array[@]}"; do
          if [ -f "$file" ]; then
            file_digests+="FILE: $file"$'\n'
            file_digests+="$(head -n 20 "$file")"$'\n---\n'
          fi
        done
        gemini_prompt="${gemini_prompt//FILE_DIGESTS/$file_digests}"
        random_files="${random_files_array[*]}"
      fi
    fi
    echo "Files to improve: $random_files"
    repomix --include "./content/**/*.md" --style markdown --output combined.md
    gemini --model gemini-2.5-pro --prompt "$gemini_prompt" > "$instructions_file"
    jq empty "$instructions_file" 2>/dev/null || { echo "Invalid JSON from Gemini"; return 1; }
    run_and_sleep "quartz_research" quartz_research "Follow the instructions in $instructions_file. Verify before running."
  done
}

# Quartz task queue
export QUARTZ_QUEUE_DIR="$HOME/.quartz_research_queue"
export QUARTZ_QUEUE_FILE="$QUARTZ_QUEUE_DIR/queue.txt"
export QUARTZ_QUEUE_LOCK="$QUARTZ_QUEUE_DIR/lock"
export QUARTZ_QUEUE_PID="$QUARTZ_QUEUE_DIR/worker.pid"
mkdir -p "$QUARTZ_QUEUE_DIR"

_quartz_research_worker() {
  while true; do
    local job
    job=$(flock -x "$QUARTZ_QUEUE_LOCK" -c '
      if [ -s "$QUARTZ_QUEUE_FILE" ]; then
        head -n1 "$QUARTZ_QUEUE_FILE"
        tail -n +2 "$QUARTZ_QUEUE_FILE" > "$QUARTZ_QUEUE_FILE.tmp"
        mv "$QUARTZ_QUEUE_FILE.tmp" "$QUARTZ_QUEUE_FILE"
      fi
    ')
    [[ -z $job ]] && { rm -f "$QUARTZ_QUEUE_PID"; exit 0; }
    local dir=${job%%|*} topic=${job#*|}
    ( cd "$dir" && quartz_research "$topic" )
  done
}

quartz_research_queue() {
  [[ $# -eq 0 ]] && { echo "Usage: quartz_research_queue <topic1> [topic2] ..."; return 1; }
  local dir="$PWD"
  for topic in "$@"; do printf '%s|%s\n' "$dir" "$topic" >> "$QUARTZ_QUEUE_FILE"; done
  echo "🗂️  Queued: $*"
  if ! { [ -f "$QUARTZ_QUEUE_PID" ] && kill -0 "$(cat "$QUARTZ_QUEUE_PID")" 2>/dev/null; }; then
    _quartz_research_worker &!
    echo $! > "$QUARTZ_QUEUE_PID"
    echo "⏳ Worker started (PID $(cat "$QUARTZ_QUEUE_PID"))."
  fi
}
quartz_queue_status(){ echo "📋 Pending jobs:"; awk -F'|' '{printf "  • %s — %s\n",$1,$2}' "$QUARTZ_QUEUE_FILE" 2>/dev/null || echo "  (empty)"; }
quartz_queue_cancel_all(){ : > "$QUARTZ_QUEUE_FILE" && echo "🚮 Cleared the queue."; }

#### 13) Deepwiki course builder (kept)
course_builder_plan(){
  local MCP_CONFIG=$(cat <<'EOF'
{
  "mcpServers": {
    "deepwiki": { "type": "sse", "url": "https://mcp.deepwiki.com/sse" }
  }
}
EOF
)
  local prompt=$(cat <<'EOF'
# Create a plan from the user's request
## Prepare
Use mcp__deepwiki__ask_question in "badass-courses/course-builder" repo.
## Plan Format
- Markdown; "goal" as H1
- Each step as a commit message
- Verification steps; exact file paths and sections; explicit changes
EOF
)
  cl --system-prompt "$prompt" "$@"
}
course_builder_claude(){ claude "$(ask_course_builder "$@")"; }

#### 14) Container-use/Claude harness
# (kept, with prompt appended to full user input)
cl(){
  local system_prompt=$(cat <<'EOF'
ALWAYS use ONLY Environments for ANY file/code/shell ops—NO EXCEPTIONS.
Do NOT install/use git cli via environment_run_cmd; use env tools.
Tell the user: container-use log <env_id> and container-use checkout <env_id>.
EOF
)
  local allowed_tools=$(cat <<'EOF' | tr '\n' ',' | sed 's/,$//'
mcp__container-use__environment_checkpoint
mcp__container-use__environment_create
mcp__container-use__environment_add_service
mcp__container-use__environment_file_delete
mcp__container-use__environment_file_list
mcp__container-use__environment_file_read
mcp__container-use__environment_file_write
mcp__container-use__environment_open
mcp__container-use__environment_run_cmd
mcp__container-use__environment_update
Web
EOF
)
  local append_user_prompt=$(cat <<'EOF'
Before you begin:
1) use container-use to run commands
2) use conventional commit messages
EOF
)
  trap 'claude mcp remove container-use 2>/dev/null || true' EXIT ERR
  claude mcp add container-use -- container-use stdio || echo "container-use already installed"
  local improved_prompt="$*\n\n$append_user_prompt"
  claude --allowedTools "$allowed_tools" \
    --dangerously-skip-permissions \
    --append-system-prompt "$system_prompt" \
    "$improved_prompt"
}

#### 15) Prompt tooling
improve(){
  claude --append-system-prompt "$(cat ~/.claude/prompts/improve.md)" "$@"
}

gist(){ claude --print --settings "$HOME/.claude-settings/gist.json" "Create a gist of the following: $@"; }
github_tasks(){ claude --settings "$HOME/.claude-settings/github-tasks.json" "$@"; }
create_repo(){ local args="${@:-Initialize a repository in: $(pwd)}"; claude --settings "$HOME/.claude-settings/repoinit.json" "$args"; }

# Duplicates removed: keep a single popus()
dopus(){ claude --dangerously-skip-permissions "$@"; }
popus(){ dopus "$(pbpaste) --- $@"; }
copus(){ claude --dangerously-skip-permissions "$@" --continue; }
conpus(){ claude --allowedTools mcp__container-use__environment_checkpoint,mcp__container-use__environment_create,mcp__container-use__environment_add_service,mcp__container-use__environment_file_delete,mcp__container-use__environment_file_list,mcp__container-use__environment_file_read,mcp__container-use__environment_file_write,mcp__container-use__environment_open,mcp__container-use__environment_run_cmd,mcp__container-use__environment_update --dangerously-skip-permissions "$@"; }

export PATH="$PATH:$HOME/dev/claude-workshop-live/bin:$HOME/dev/claude-workshop-live/jq-filters" # already in array; harmless

#### 16) API key helpers (1Password)
with_github(){ local k; k= "**********"="$k" "$@"; }
zai(){ local k; k=$(op item get "ZAI_API_KEY" --fields credential --reveal | tr -d '\n') || { echo "No ZAI key" >&2; return 1; }
  ANTHROPIC_AUTH_TOKEN="$k" ANTHROPIC_BASE_URL="https: "**********"
kimi(){
  local k
  k=$(op item get "OPENROUTER_KIMI_API_KEY" --fields credential --reveal | tr -d '\n') || { echo "No Kimi key" >&2; return 1; }

  __kimi_with_router(){
    local previous_key="$OPENROUTER_KIMI_API_KEY"
    export OPENROUTER_KIMI_API_KEY="$k"

    ccr code "$@"
    local status=$?

    if [[ -n "$previous_key" ]]; then
      export OPENROUTER_KIMI_API_KEY="$previous_key"
    else
      unset OPENROUTER_KIMI_API_KEY
    fi

    return $status
  }

  __kimi_with_router "$@"
  local exit_code=$?
  unset -f __kimi_with_router
  return $exit_code
}
with_free_gemini(){ local k; k=$(op item get "GEMINI_API_KEY_FREE" --fields credential --reveal | tr -d '\n') || { echo "No free Gemini key" >&2; return 1; }; GEMINI_API_KEY="$k" "$@"; }
with_zai(){ local k; k=$(op item get "ZAI_API_KEY" --fields credential --reveal | tr -d '\n') || { echo "No ZAI key" >&2; return 1; }
  ANTHROPIC_AUTH_TOKEN="$k" ANTHROPIC_BASE_URL="https: "**********"
with_zai_key_only(){ local k; k=$(op item get "ZAI_API_KEY" --fields credential --reveal | tr -d '\n') || { echo "No ZAI key" >&2; return 1; }; ZAI_API_KEY="$k" "$@"; }
with_gemini(){ local k; k=$(op item get "GEMINI_API_KEY" --fields credential --reveal | tr -d '\n') || { echo "No Gemini key" >&2; return 1; }; GEMINI_API_KEY="$k" "$@"; }
with_openai(){ local k; k=$(op item get "OPENAI_API_KEY" --fields credential --reveal | tr -d '\n') || { echo "No OpenAI key" >&2; return 1; }; OPENAI_API_KEY="$k" "$@"; }

vid(){ with_gemini claude-video "$@"; }

#### 17) Env runner
with(){ [[ -f .env ]] || { echo "No .env in $(pwd)"; return 1; }; (export $(grep -v '^#' .env | xargs) && "$@"); }
nn8n(){ NODE_FUNCTION_ALLOW_BUILTIN=* n8n "$@"; }

graude(){
  export ANTHROPIC_BASE_URL="https://api.groq.com/openai/chat/completions"
  export ANTHROPIC_AUTH_TOKEN= "**********"
  export ANTHROPIC_MODEL="openai/gpt-oss-120b"
  claude "$@"
}

create_cred(){ local title="$1" credential="$2"; op item create --category "API Credential" --title "$title" credential="$credential"; }

#### 18) App scaffolding
next_app(){
  local target_dir="${1:-.}"
  yes '' | pnpm create next-app@latest "$target_dir" \
    --tailwind --biome --typescript --app --no-src-dir --import-alias "@/*" --overwrite \
  && echo "✅ Next.js app created in '$target_dir'" || { echo "❌ Failed to create Next.js app"; return 1; }
}

gem(){ with_gemini gemsum "$@"; }
add_bm(){ local project_name="$(basename "$PWD")"; bm project add "$project_name" .; claude mcp add -t stdio basic-memory -- bm --project "$project_name/memory" mcp; }

# File / command helpers
filei(){ { cat "$1"; echo "$2"; } | claude; }
filep(){ { cat "$1"; echo "$2"; } | claude --print; }
cmdi(){ { eval "$1"; echo "$2"; } | claude; }
cmdp(){ { eval "$1"; echo "$2"; } | claude --print; }

dopex(){ codex --dangerously-bypass-approvals-and-sandbox -c model_reasoning_effort=high "$@"; }
upai(){ bun i -g @openai/codex@latest @anthropic-ai/claude-code@latest @google/gemini-cli@latest @github/copilot@latest; }

codex_continue(){
  local latest
  latest=$(find ~/.codex/sessions -type f -name '*.jsonl' -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)
  [[ -z "$latest" ]] && { echo "No codex sessions found"; return 1; }
  echo "Resuming from: $latest"
  codex --config experimental_resume="$latest" "$@"
}

cload(){ local context=$(find ai -type f -name "*.md" -exec cat {} \;); claude --append-system-prompt "$context" "$@"; }
backlog_next(){ dopus "Read the next task from the backlog. Follow git flow best practices and create a branch, work the task, then commit/push/PR using gh."; }

learn(){ claude --settings '{"outputStyle": "interactive-doc-learner", "permissions": {"allow": ["Bash(curl -sL \"https://into.md:*\")"]}}' "$@"; }
catwithnames(){ for f in "$@"; do echo "=== $f ==="; cat "$f"; done; }

imagegen(){ gemini "/generate $@" -y; }
imagegen_clipboard(){ gemini "Use the /generate command to generate: <image_prompt> $@ </image_prompt>; then copy the image to the clipboard" -y; }
gemini_plan(){ local input="$@"; [ -p /dev/stdin ] && input="$(cat) $input"; gemini "Create and output a plan for: $input"; }

claude_qa(){ claude --append-system-prompt "$(cat qa.md)" "$@"; }

# Keep a single, unshadowed commit helper
commit(){ claude "/conventional-commit" "$@" --print --verbose; }

claude2gemini(){
  claude --print --output-format json --max-turns 1 "$@" \
  | jq -r '.text | gsub("\\s+"; " ")' \
  | gemini -y "Use the /generate command to create this image"
}

# Custom functions registry (user file)
source ~/.zsh/custom/functions.zsh

# bun completions
[ -s "$HOME/.bun/_bun" ] && source "$HOME/.bun/_bun"


# Add to ~/.bashrc or ~/.zshrc
# Add to ~/.bashrc or ~/.zshrc
ob() {
     # Point to the memory subdirectory
    local memory_path="$PWD/memory"
    
    if [ ! -d "$memory_path" ]; then
        echo "Memory folder not found at $memory_path"
        return 1
    fi
    
    # Create .obsidian folder if it doesn't exist
    if [ ! -d "$memory_path/.obsidian" ]; then
        echo "Initializing Obsidian vault in memory folder..."
        mkdir -p "$memory_path/.obsidian"
        echo '{"cssTheme":""}' > "$memory_path/.obsidian/appearance.json"
    fi
    
    # Open in Obsidian
    open "obsidian://open?path=$memory_path"
}

unalias ghv 2>/dev/null
ghv() { gh repo view --web "$@"; }

unalias ghcp 2>/dev/null
ghcp() { GH_PAGER="" gh repo view --json name,owner --jq "\"https://github.com/\" + .owner.login + \"/\" + .name" | pbcopy; }

unalias mcpi 2>/dev/null
mcpi() { bunx @modelcontextprotocol/inspector@latest "$@"; }

unalias h 2>/dev/null
h() { claude --model haiku "$@"; }

unalias x 2>/dev/null
x() { claude --dangerously-skip-permissions "$@"; }
dex(){
  codex --dangerously-bypass-approvals-and-sandbox
}
xcon() { claude --dangerously-skip-permissions --continue "$@"; }
xres() { claude --dangerously-skip-permissions --resume "$@"; }

unalias cc 2>/dev/null
cc() { claude "$@"; }

unalias cdi 2>/dev/null
cdi() {
  claude --append-system-prompt "$(files ai/diagrams/**/*.md)"
}


inspiration(){
  claude --system-prompt "You are an inspirational speaker who comes up with 5 ideas for the user's topic" "$@"
}

#### 19) Programming Persona Functions
# Each persona embodies specific programming philosophies and provides opinionated feedback
# All personas use the Skill(review) pattern for consistency and reference SKILL.md

PERSONA_EXPERT_PREFIX="Run the Skill(review) tool with the"

persona-new(){
  claude --system-prompt "<example-personas>$(files /Users/johnlindquist/.claude/skills/review/references/*.md)</example-personas> --- You are an expert in creating new persona markdown files following the same format as the examples. Create a new persona markdown file for the user's request and add it to '/Users/johnlindquist/.claude/skills/review/references'" "$@"
}

persona-linus() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX linus persona. Run the Skill(review) with the reference file references/linus-reviewer.md" "$@"
}

persona-guido() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX guido persona. You are Guido van Rossum. You personify the ideals of clarity, simplicity, and code readability. Fully embrace these ideals and push back against unnecessary complexity, clever one-liners, or inconsistent style." "$@"
}

persona-brendan() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX brendan persona. You are Brendan Eich. You personify the ideals of rapid innovation, adaptability, and creative problem-solving under pressure. Fully embrace these ideals and push back against slow, dogmatic development or lack of experimentation." "$@"
}

persona-bjarne() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX bjarne persona. You are Bjarne Stroustrup. You personify the ideals of performance through abstraction, type safety, and disciplined engineering. Fully embrace these ideals and push back when people trade performance for convenience or forget design integrity." "$@"
}

persona-james() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX james persona. You are James Gosling. You personify the ideals of platform independence, reliability, and scalability. Fully embrace these ideals and push back against language fragmentation, sloppy deployment, or non-portable code." "$@"
}

persona-anders() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX anders persona. You are Anders Hejlsberg. You personify the ideals of strong typing, developer productivity, and elegant tooling. Fully embrace these ideals and push back against dynamic chaos, weak tooling, or lack of structure." "$@"
}

persona-unix() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX unix persona. You are Ken Thompson and Dennis Ritchie combined. You personify the ideals of minimalism, composability, and doing one thing well. Fully embrace these ideals and push back against bloat, abstraction for its own sake, or unnecessary frameworks." "$@"
}

persona-rob() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX rob persona. You are Rob Pike. You personify the ideals of simplicity, concurrency, and clarity in design. Fully embrace these ideals and push back on excessive abstraction, verbosity, or anything that adds friction to problem-solving." "$@"
}

persona-matz() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX matz persona. You are Yukihiro \"Matz\" Matsumoto. You personify the ideals of developer happiness, elegant design, and humane code. Fully embrace these ideals and push back when efficiency or convention trumps joy, flow, or creativity." "$@"
}

persona-dhh() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX dhh persona. You are David Heinemeier Hansson. You personify the ideals of opinionated software, developer autonomy, and simplicity through convention. Fully embrace these ideals and push back hard against unnecessary configuration, corporate overengineering, or process obsession." "$@"
}

persona-fowler() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX fowler persona. You are Martin Fowler. You personify the ideals of refactoring, maintainability, and evolving architecture. Fully embrace these ideals and push back on big rewrites, tech fads, and architecture without purpose." "$@"
}

persona-beck() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX beck persona. You are Kent Beck. You personify the ideals of test-driven development, feedback cycles, and adaptive design. Fully embrace these ideals and push back against untested code, fear-driven engineering, or planning without iteration." "$@"
}

persona-grace() {
  claude --system-prompt "Load the $PERSONA_EXPERT_PREFIX using the grace persona." "$@"
}

persona-carmack() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX carmack persona. You are John Carmack. You personify the ideals of low-level excellence, performance optimization, and precision thinking. Fully embrace these ideals and push back hard on hand-waving, inefficiency, or lack of technical depth." "$@"
}

persona-dean() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX dean persona. You are Jeff Dean. You personify the ideals of scale, efficiency, and practical genius. Fully embrace these ideals and push back on theoretical fluff, poor infrastructure design, or wasteful computation." "$@"
}

persona-github() {
  claude --system-prompt "You are an expert of the Skill(github) tool with the the github persona. You are the GitHub Generation. You personify the ideals of collaboration, transparency, and continuous integration. Fully embrace these ideals and push back when contributions are siloed, undocumented, or not shared back with the community." "$@"
}

persona-perf() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX perf persona. You are Brendan Gregg (and Liz Rice's spirit). You personify the ideals of systems observability, real-world performance analysis, and deep tooling literacy. Fully embrace these ideals and push back against shallow metrics, guesswork debugging, or hidden complexity." "$@"
}

persona-lattner() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX lattner persona. You are Chris Lattner. You personify the ideals of language infrastructure, interoperability, and compiler craftsmanship. Fully embrace these ideals and push back against reinventing wheels or building systems without reusable cores." "$@"
}

persona-react() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX react persona. You are Jordan Walke and Dan Abramov merged. You personify the ideals of declarative UI, functional design, and state predictability. Fully embrace these ideals and push back when code mutates state chaotically or lacks a clear data flow." "$@"
}

persona-ai() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX ai persona. You are the AI Visionaries — Karpathy, Howard, Chollet, and Hassabis unified. You personify the ideals of self-learning systems, code that adapts, and the fusion of reasoning with computation. Fully embrace these ideals and push back when design thinking ignores data, feedback, or emergent behavior." "$@"
}

# bat injects files names
alias cat='bat --no-pager'


files(){
  for file in "$@"; do
    [ -d "$file" ] && continue
    echo "=== $file ==="
    cat "$file"
    echo
  done
}

plan(){
  local files=(ai/diagrams/**/*.md(N))
  if [[ ${#files[@]} -gt 0 ]]; then
    local prompt_content="<diagrams>$(files ${files[@]})</diagrams>"
    claude --permission-mode "plan" --append-system-prompt "$prompt_content" "$@"
  else
    claude --permission-mode "plan" "$@"
  fi
}


# plan(){
#   local files=(ai/diagrams/**/*.md(N))
#   if [[ ${#files[@]} -gt 0 ]]; then
#     local prompt_content="$(cat ${files[@]} 2>/dev/null)"
#     claude --permission-mode "plan" --append-system-prompt "$prompt_content" "$@"
#   else
#     claude --permission-mode "plan" "$@"
#   fi
# }

github-issue-create(){
  claude --settings "$HOME/.claude/settings/settings.github.json" --system-prompt "Load the Skill(github) and load the referenced CREATE_ISSUE.md file to create an issue for the following" "$@"
}


persona-react-typescript() {
  claude --system-prompt "$PERSONA_EXPERT_PREFIX react and typescript personas" "$@"
}

refactor-react-typescript(){
  # assign a variable to the persona-react-typescript output
  local output=$(persona-react-typescript "$@" --print)
  dopus "Follow the instructions in this refactor plan: $output"
}

github-issue-create-react-typescript(){
  # assign a variable to the persona-react-typescript output
  local output=$(persona-react-typescript "$@" --print)
  echo "Issue: $output"
  github-issue-create "$output"  
}

diagram-create(){
  claude --allowed-tools "Skill(diagram)" --system-prompt "You are an expert of the Skill(diagram) tool. You create diagrams based on the user's prompt." "$@"
}

chrome-devtools(){
  local _settings="$HOME/.claude/settings/settings.chrome-devtools.json"
  local _mcp_config="$HOME/.claude/mcp/mcp.chrome-devtools.json"


  echo "$_settings"
  echo "$_mcp_config"

  claude --settings "$_settings" --mcp-config "$_mcp_config" --system-prompt "You are an expert of the Skill(chrome-devtools) tool. Connect to the given URL and ask the user what they want to do." "$@"
}

alias cat='bat --no-pager'
alias ls='eza'

clone-and-check(){
  gh repo c
}

binit(){
  bun init --yes
}

hooks-init(){
  mkdir -p .claude/hooks
  # capture the current directory
  local current_dir=$(pwd)
  # set cwd to .claude/hooks
  cd .claude/hooks
  bun init --yes
  bun i @anthropic-ai/claude-agent-sdk
  cat <<'EOF' > index.ts
import type { } from "@anthropic-ai/claude-agent-sdk"

const input = await Bun.stdin.json();
EOF
  # set cwd to the current directory
  cd $current_dir
  cursor .claude/hooks/index.ts
  claude --model haiku "/hooks"
}



cont(){
  claude --continue "$@"
}

resu(){
  claude --resume "$@"
}

brainpick(){
  claude \
  --setting-sources "" \
  --settings '{"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "echo \"Remember to always use the AskUserQuestion tool.\""}]}]}}'\
  --model haiku \
  --system-prompt "You are an expert in helping the user clarify their intentions by asking thoughtful, targeted questions.
Your goal is to surface the user's underlying ideas, goals, and preferences—not to decide for them.
Let the user think. Ask one question at a time. Guide, don't lead." \
  "$@"
}

# Put this in your ~/.zshrc

# Drop this into your ~/.zshrc
muxai() {
  local session="${1:-ai}"     # session name when launching from outside tmux
  local window="${2:-bots}"    # window name

  local -a cmds=(claude gemini ca copilot codex droid)

  # Require tmux
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found in PATH."
    return 1
  fi

  local win_id pane_id win_info

  if [[ -n "$TMUX" ]]; then
    # Already in tmux: make a new window in the current session
    win_info=$(tmux new-window -P -F "#{window_id}|#{pane_id}" -n "$window")
    win_id="${win_info%%|*}"
    pane_id="${win_info##*|}"
  else
    # Not in tmux: use/create the requested session
    if tmux has-session -t "$session" 2>/dev/null; then
      win_info=$(tmux new-window -t "$session" -P -F "#{window_id}|#{pane_id}" -n "$window")
      win_id="${win_info%%|*}"
      pane_id="${win_info##*|}"
    else
      tmux new-session -d -s "$session" -n "$window"
      win_id=$(tmux display-message -p -t "$session:$window" "#{window_id}")
      pane_id=$(tmux list-panes     -t "$session:$window" -F "#{pane_id}" | head -n1)
    fi
  fi

  # Keep panes open if a command exits quickly (useful for errors)
  tmux set-window-option -t "$win_id" remain-on-exit on >/dev/null

  # First command goes in the original pane
  tmux send-keys -t "$pane_id" "${cmds[1]}" C-m
  tmux select-pane -t "$pane_id" -T "${cmds[1]}" 2>/dev/null || true

  # Split for the rest; alternate split directions, then tile
  local current="$pane_id"
  local new_pane
  local i
  for (( i=2; i<=${#cmds[@]}; i++ )); do
    if (( i % 2 == 0 )); then
      new_pane=$(tmux split-window -P -F "#{pane_id}" -t "$current" -h)
    else
      new_pane=$(tmux split-window -P -F "#{pane_id}" -t "$current" -v)
    fi
    tmux send-keys -t "$new_pane" "${cmds[i]}" C-m
    tmux select-pane -t "$new_pane" -T "${cmds[i]}" 2>/dev/null || true
    current="$new_pane"
  done

  tmux select-layout -t "$win_id" tiled >/dev/null
  tmux rename-window -t "$win_id" "$window" 2>/dev/null || true
  tmux select-window -t "$win_id"

  # If we started outside tmux, attach now
  if [[ -z "$TMUX" ]]; then
    tmux attach -t "$session"
  fi
}


tmux source-file ~/.tmux.conf



slugify(){
  tr '[:upper:]' '[:lower:]' | tr -d '[:punct:]' | tr ' ' '-' | tr '/' '-'
}

news(){
  local slugified_query=$(echo "$@" | slugify)
  local file_name="$(date +"%Y-%m-%d-%H-%M")-$slugified_query.md"
  local output_file="$(pwd)/$file_name"
  local system_prompt=$(cat <<EOF
You're an expert researcher and summarizer on the latest news. 
EOF
)

  local prompt=$(cat <<EOF
## Critical Steps
Step 1: Research: "$@"
Step 2: Write a summary to $file_name
EOF
)
  local allowed_tools="WebSearch,WebFetch,Write(./$file_name),Read(./$file_name)"
  echo "system_prompt: $system_prompt"
  echo "allowed_tools: $allowed_tools"
  echo "output_file: $output_file"
  echo "prompt: $prompt"
  claude --setting-sources="" --model=haiku --system-prompt="$system_prompt" --allowedTools="$allowed_tools" "$prompt"
}


test-write(){
#   local settings=$(cat <<EOF
# {
#   "permissions": {
#     "allow": [
#       "Write($(pwd)/test-write.md)"
#     ],
#     "deny": [],
#     "ask": []
#   }
# } 
# EOF
# )
  echo "settings: $settings"
  claude --setting-sources="" --allowed-tools="Write(./test-write-4.md)" --disallowed-tools="Read" --model=haiku --system-prompt="You are an expert writer. You write a summary to a file." "Write a summary of how the sun shines to ./test-write-4.md"
}

best-practices(){
  local slugified_query=$(echo "$@" | slugify)
  local current_date=$(date +"%Y-%m-%d")
  local file_name="$(date +"%Y-%m-%d-%H-%M")-$slugified_query.md"
  local output_file="$(pwd)/$file_name"
  local system_prompt=$(cat <<EOF
You're an expert in researching the latest coding best practices. 
EOF
)

  local prompt=$(cat <<EOF
The current date is $current_date. We're looking for the absolute latest best practices for the following files: "$@".

Ignore any advice that is over a year old.

## Critical Steps
Step 1: Using only the WebSearch and WebFetch tools, research the best practices for the files listed here: "$@".
Step 2: Write a summary to $file_name
EOF
)
  local allowed_tools="WebSearch,WebFetch"
  echo "system_prompt: $system_prompt"
  echo "allowed_tools: $allowed_tools"
  echo "output_file: $output_file"
  echo "prompt: $prompt"
  claude --setting-sources="" --model=haiku --system-prompt="$system_prompt" --allowedTools="$allowed_tools" "$prompt"
}
# Added by Antigravity
export PATH="/Users/johnlindquist/.antigravity/antigravity/bin:$PATH"

# =============================================================================
# 👻 Git Ghost Checkpoints
# =============================================================================
# A workflow for saving "invisible" commits (refs/ghosts/) that don't pollute
# your branch history or staging area. Includes auto-cleanup.

GHOST_REF_PREFIX="refs/ghosts"

# 1. SAVE
# Snapshots current state (staged+unstaged) to a hidden ref.
# Usage: ghost-save "optional message"
function ghost-save() {
    local msg="${1:-Ghost Checkpoint}"
    local ts=$(date +%s)
    
    [ -d .git ] || { echo "❌ Not a git repo."; return 1; }

    # Create a temp index to avoid touching the user's actual index
    local temp_index=".git/index.ghost.$ts"
    trap 'rm -f "$temp_index"' EXIT

    # Copy current index to temp, then add all working changes
    cp .git/index "$temp_index" 2>/dev/null
    GIT_INDEX_FILE="$temp_index" git add -A

    # Write the tree object
    local tree=$(GIT_INDEX_FILE="$temp_index" git write-tree)
    local parent=$(git rev-parse HEAD)

    # DEDUPLICATION: Check if this tree matches the last ghost
    local last_ghost=$(git for-each-ref --sort=-committerdate --count=1 --format='%(objectname)' "$GHOST_REF_PREFIX")
    if [[ -n "$last_ghost" ]]; then
        local last_tree=$(git rev-parse "$last_ghost^{tree}")
        if [[ "$last_tree" == "$tree" ]]; then
            _ghost_autoclean # Still run cleanup even if we skip save
            return 0
        fi
    fi

    # Commit the tree (plumbing)
    local commit=$(echo "$msg" | git commit-tree "$tree" -p "$parent")
    git update-ref "$GHOST_REF_PREFIX/$ts" "$commit"
    echo "👻 Saved ghost: ${commit:0:7}"

    # Trigger auto-cleanup of old ghosts
    _ghost_autoclean
}

# 2. LOG
# Lists the last 20 ghost checkpoints.
# Usage: ghost-log
function ghost-log() {
    git for-each-ref --sort=-committerdate "$GHOST_REF_PREFIX/" \
    --format='%(color:yellow)%(refname:short)%(color:reset) | %(color:green)%(objectname:short)%(color:reset) | %(contents:subject) %(color:blue)(%(committerdate:relative))%(color:reset)' \
    | sed "s|$GHOST_REF_PREFIX/||" | head -n 20
}

# 3. RESTORE
# Hard resets working directory to a specific ghost state.
# Usage: ghost-restore <timestamp_id> [--force]
function ghost-restore() {
    local id="$1"
    local force="$2"

    [ -z "$id" ] && { echo "Usage: ghost-restore <timestamp_id> [--force]"; return 1; }

    # SAFETY: Prevent overwriting uncommitted work
    if [[ -n $(git status --porcelain) ]] && [[ "$force" != "--force" ]]; then
        echo "⚠️  Working directory is dirty!"
        echo "   Restoring will overwrite your work. Use 'ghost-restore $id --force'."
        return 1
    fi
    
    local ref="$GHOST_REF_PREFIX/$id"
    if ! git show-ref --quiet "$ref"; then
        echo "❌ Ghost $id not found."
        return 1
    fi

    echo "Rewinding to $id..."
    git read-tree --reset -u "$ref"
}

# 4. EXPORT
# Zips a specific ghost state for sharing/archiving.
# Usage: ghost-export <timestamp_id>
function ghost-export() {
    local id="$1"
    [ -z "$id" ] && { echo "Usage: ghost-export <timestamp_id>"; return 1; }

    local ref="$GHOST_REF_PREFIX/$id"
    if ! git show-ref --quiet "$ref"; then
        echo "❌ Ghost $id not found."
        return 1
    fi

    local filename="ghost-export-${id}.zip"
    git archive --format=zip --output="$filename" "$ref"
    echo "📦 Exported ghost $id to ./$filename"
}

# 5. NUKE
# Deletes all ghosts immediately.
# Usage: ghost-nuke
function ghost-nuke() {
    git for-each-ref --format='%(refname)' "$GHOST_REF_PREFIX/" | xargs -L1 git update-ref -d
    echo "💥 All ghosts busted."
}

# INTERNAL: AUTO-CLEANUP
# Moves ghosts > 7 days old to /tmp bundles and deletes refs.
function _ghost_autoclean() {
    local cutoff=$(($(date +%s) - 604800)) # 7 days
    local graveyard="${TMPDIR:-/tmp}/ghost-graveyard"
    
    # Check for refs older than cutoff
    git for-each-ref --format='%(refname) %(committerdate:unix)' "$GHOST_REF_PREFIX" \
    | while read ref timestamp; do
        if [[ "$timestamp" -lt "$cutoff" ]]; then
            mkdir -p "$graveyard"
            local short_id=${ref#$GHOST_REF_PREFIX/}
            local bundle_path="$graveyard/ghost-${short_id}.bundle"
            
            # Create git bundle (archive) and delete ref
            git bundle create "$bundle_path" "$ref" >/dev/null 2>&1
            git update-ref -d "$ref"
            # echo "⚰️  Buried old ghost $short_id" # Uncomment if you want noise
        fi
    done
}

af(){
  a --force "$@"
}



claude-demo(){
  claude "Write a poem about the following: $@"
}ref#$GHOST_REF_PREFIX/}
            local bundle_path="$graveyard/ghost-${short_id}.bundle"
            
            # Create git bundle (archive) and delete ref
            git bundle create "$bundle_path" "$ref" >/dev/null 2>&1
            git update-ref -d "$ref"
            # echo "⚰️  Buried old ghost $short_id" # Uncomment if you want noise
        fi
    done
}

af(){
  a --force "$@"
}



claude-demo(){
  claude "Write a poem about the following: $@"
}