#date: 2021-12-16T17:05:48Z
#url: https://api.github.com/gists/c8ab83bf5e087dbae86796289e430234
#owner: https://api.github.com/users/nigelgbanks

#!/usr/bin/env bash
set -e

readonly PROGNAME=$(basename $0)
readonly PROGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ARGS="$@"

readonly LATEST_VERSION="2.16.0"

function usage() {
  cat <<-EOF
    usage: $PROGNAME

    Modifies log4j jars to prevent security issues:
    - <https://nvd.nist.gov/vuln/detail/CVE-2021-44228>
    - <https://nvd.nist.gov/vuln/detail/CVE-2019-17571>

    OPTIONS:
      -d --dry-run       Do not modify the system.
      -y --yes           Do not prompt to continue.
      -h --help          Show this help.
      -x --debug         Debug this script.

    Examples:
      Describe what the script would do.
      $PROGNAME -d
EOF
}

function cmdline() {
  local arg=
  for arg; do
    local delim=""
    case "$arg" in
    # Translate --gnu-long-options to -g (short options)
    --dry-run) args="${args}-d " ;;
    --yes) args="${args}-y " ;;
    --help) args="${args}-h " ;;
    --debug) args="${args}-x " ;;
    # Pass through anything else
    *)
      [[ "${arg:0:1}" == "-" ]] || delim="\""
      args="${args}${delim}${arg}${delim} "
      ;;
    esac
  done

  # Reset the positional parameters to the short options
  eval set -- $args

  while getopts "dyhx" OPTION; do
    case $OPTION in
    d)
      readonly DRY_RUN="true"
      ;;
    y)
      readonly NO_PROMPT="true"
      ;;
    h)
      usage
      exit 0
      ;;
    x)
      readonly DEBUG='-x'
      set -x
      ;;
    esac
  done

  if [[ -z $DRY_RUN ]]; then
    readonly DRY_RUN="false"
  fi

  if [[ -z $NO_PROMPT ]]; then
    readonly NO_PROMPT="false"
  fi

  return 0
}

function has_cmd {
  local cmd="${1}"
  which "${cmd}" &>/dev/null
}

function install {
  local package="${1}"
  local sudo_cmd=""
  if has_cmd "sudo"; then
    sudo_cmd="sudo"
  fi
  if not_dry_run; then
    if has_cmd "apk"; then
      ${sudo_cmd} apk add "${package}"
    elif has_cmd "apt"; then
      ${sudo_cmd} apt-get install "${package}"
    elif has_cmd "yum"; then
      ${sudo_cmd} yum install "${package}"
    fi
  fi
}

function prerequisites {
  local grep=$(basename $(readlink -f $(which grep)))
  log 1 "Prerequisites"
  log 2 "Require Zip"
  if ! has_cmd "zip"; then
    install "zip"
  else
    log 2 "Found Zip"
  fi
  if [[ "$grep" == "busybox" ]]; then
    log 2 "Require GNU grep"
    install "grep"
  else
    log 2 "Has GNU grep"
  fi
}

# Stolen from <https://stackoverflow.com/a/4025065/1558186>
function vercomp {
  if [[ $1 == $2 ]]; then
    return 0
  fi
  local IFS=.
  local i ver1=($1) ver2=($2)
  # fill empty fields in ver1 with zeros
  for ((i = ${#ver1[@]}; i < ${#ver2[@]}; i++)); do
    ver1[i]=0
  done
  for ((i = 0; i < ${#ver1[@]}; i++)); do
    if [[ -z ${ver2[i]} ]]; then
      # fill empty fields in ver2 with zeros
      ver2[i]=0
    fi
    if ((10#${ver1[i]} > 10#${ver2[i]})); then
      return 1
    fi
    if ((10#${ver1[i]} < 10#${ver2[i]})); then
      return 2
    fi
  done
  return 0
}

function not_dry_run {
  [[ "${DRY_RUN}" == "false" ]]
}

function should_prompt {
  [[ "${NO_PROMPT}" == "false" ]]
}

function log {
  local indent="${1}"
  shift
  local msg="${@}"
  for ((i = 1; i < $indent; i++)); do
    echo -n $'  '
  done
  echo "${msg}"
}

function prompt_exit {
  if should_prompt; then
    read -p 'Would you like to continue (y/n): ' prompt
    if [[ "${prompt}" != "y" ]]; then
      exit 1
    fi
  fi
}

function has_class {
  local jar="${1}"
  local class="${2}"
  if not_dry_run; then
    zip -d "${jar}" "${class}"
  fi
}

function remove_class {
  local jar="${1}"
  local class="${2}"
  readarray -d '' instances < <(unzip -Z -1 "${jar}" | grep -Z "${class}" | tr "\n" "\0")
  if [ ${#instances[@]} -eq 0 ]; then
    log 3 "Missing ${class}"
    return
  fi
  for instance in "${instances[@]}"; do
    log 3 "Removed ${instance} from ${jar}"
    log 4 zip -qd "${jar}" "${instance}"
    if not_dry_run; then
      zip -qd "${jar}" "${instance}"
    fi
  done
}

function remove_classes {
  local jar="${1}"
  shift
  local classes=("${@}")
  for class in "${classes[@]}"; do
    remove_class "${jar}" "${class}"
  done
}

function package {
  local path="${1}"
  local name=$(basename "${path}")
  echo "${name%-*}"
}

function version {
  local path="${1}"
  local name=$(basename "${path}")
  local tmp="${name##*-}"
  echo "${tmp%.jar}"
}

function is_log4j_version_less_than_2_0 {
  local version="${1}"
  vercomp "${version}" "2.0"
  case $? in
  0) return 1 ;;
  1) return 1 ;;
  2) return 0 ;;
  esac
}

function log4j {
  local package
  local version
  classes=(
    org/apache/log4j/net/JMSAppender.class 
    org/apache/log4j/net/SocketServer.class
    org/apache/logging/log4j/core/lookup/JndiLookup.class
  )
  readarray -d '' jars < <(find / -name "*.[jw]ar" -print0 2>/dev/null)
  for jar in "${jars[@]}"; do
    log 1 "Found ${jar}"
    remove_classes "${jar}" "${classes[@]}"
  done
}

function main {
  cmdline ${ARGS}
  prerequisites
  log4j
}
main
