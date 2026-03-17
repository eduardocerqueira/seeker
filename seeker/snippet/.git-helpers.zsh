#date: 2026-03-17T17:49:53Z
#url: https://api.github.com/gists/ea9d139ba9d3e3ba039d7c98637d57ac
#owner: https://api.github.com/users/diogobas

unalias gcl 2> /dev/null
unalias gco 2> /dev/null
unalias gpu 2> /dev/null
unalias gg 2> /dev/null

alias ginit='git init; git commit --allow-empty -m "initial commit";'

function pr() {
    local branch=$(git rev-parse --abbrev-ref HEAD)
    local upstream="${1:-`git_upstream_branch`}"
    local remote="${2:-`git_find_upstream`}"
    git push "$remote" -u "$branch"
    gh pr create -B "$upstream"
}

function git_recent_branches() {
    git branch -vvv --color=always --sort=committerdate | tail | sort -r
}

function select_branch() {
    git_recent_branches | sed 's/\* /  /' | fzy | perl -pe 's/\e\[?.*?[\@-~]//g' | awk '{print $1}'
}

function gco() {
  branch=$(select_branch)
  if [ ! -z "$branch" ]; then
    git checkout "$branch"
  fi
}

function gg() {
    default=$(git_default_branch)
    git checkout "$default"
}

function git_default_branch() {
    git branch --format='%(refname:short)' | while read branch; do
      if [[ "$branch" = "dev" ]]; then
        echo "$branch"
        return
      fi
      if [[ "$branch" = "main" ]]; then
        echo "$branch"
        return
      fi
      if [[ "$branch" = "master" ]]; then
        echo "$branch"
        return
      fi
    done
    return 1
}

function gdu() {
    local branch=$(git_upstream_branch)
    if [ "$1" = "-w" ]; then
        git diff -w --ignore-blank-lines "$branch" --
    else
        git diff "$branch"
    fi
}

function gpu() {
    git push -u origin $(git rev-parse --abbrev-ref HEAD)
}

function gdb() {
  branch=$(select_branch)
  if [ ! -z "$branch" ]; then
    git branch -D "$branch"
  fi
}

function gdbs() {
    while true; do
        gdb
    done
}

function gdbm() {
  git branch --merged \
    | grep -v '*' \
    | grep -v ^master$ \
    | grep -v ^main$ \
    | grep -v ^dev$ \
    | xargs git branch -D
}

function gmv() {
  local branch=$(git rev-parse --abbrev-ref HEAD)
  local file=$(mktemp)

  echo "$branch" > "$file"
  $EDITOR "$file"
  git branch -m $(cat "$file")
  rm "$file"
}

function vimd() {
  local base="${1:-HEAD}"
  local files=$(git -c status.relativePaths=true diff --name-only "$base" -- . | xargs -I '{}' realpath --relative-to=. $(git rev-parse --show-toplevel)/'{}' | xargs)
  vim $(echo "$files")
}

function vimdr() {
  vimd $(recent_commit)
}

function nvimd() {
  local base="${1:-HEAD}"
  local files=$(git -c status.relativePaths=true diff --name-only "$base" -- . | xargs -I '{}' realpath --relative-to=. $(git rev-parse --show-toplevel)/'{}' | xargs)
  nvim $(echo "$files")
}

function nvimdr() {
  nvimd $(recent_commit)
}

function recent_commit() {
  git lg | fzy | sed 's/\x1b\[[0-9;]*m//g' | sed 's/\(\w\{7\}\).*/\1/' | awk '{print $NF}'
}

function git_find_upstream() {
  remotes=$(git remote)
  if [[ "$remotes" == *"upstream" ]]; then
    remote=upstream
  else
    remote=origin
  fi
  echo "$remote"
}

function gtd() {
  remotes=$(git remote)
  if [[ "$remotes" == *"upstream" ]]; then
    remote=upstream
  else
    remote=origin
  fi
  git push $remote $(git rev-parse --abbrev-ref HEAD):test-deploys -f
}

function gtpb() {
  remotes=$(git remote)
  if [[ "$remotes" == *"upstream" ]]; then
    remote=upstream
  else
    remote=origin
  fi
  git push $remote $(git rev-parse --abbrev-ref HEAD):test-prod-builds -f
}

function gtmpc() {
  git commit -am "temp commit" -n
  gtd
  git reset HEAD~1
}

function gawatch() {
  local branch="${1:-test-deploys}"
  gh run watch $(gh run list -b "$branch" -L 1 --json databaseId,status -q '.[0].databaseId')
}

function gcommit() {
  git commit -m "$(git rev-parse --abbrev-ref HEAD)"
}