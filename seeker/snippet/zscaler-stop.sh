#date: 2026-02-17T17:33:00Z
#url: https://api.github.com/gists/d2240b063973d5db04a07de44d79bf44
#owner: https://api.github.com/users/samwiseg0

#!/usr/bin/env bash
# vim: ai ts=2 sw=2 et sts=2 ft=sh

# Exit on error unless '|| true'.
#set -o errexit
# Exit on error inside subshells functions.
set -o errtrace
# Do not use undefined variables.
set -o nounset
# Catch errors in piped commands.
set -o pipefail

# Enable case-insensitive globbing
shopt -s nocaseglob
# Allow empty globs.
shopt -s nullglob

IFS=$' '

# Globals.
export Z_APPNAME="Zscaler"
export Z_PLUGINS="ZDP"
export Z_APP="/Applications/Zscaler/Zscaler.app"
export Z_BIN="${Z_APP}/Contents/MacOS/Zscaler"
export Z_TNL="${Z_APP}/Contents/PlugIns/ZscalerTunnel"
export Z_SRV="${Z_APP}/Contents/PlugIns/ZscalerService"

## Stop Zscaler
#
# Prevents Zscaler from being executed or restarted by removing execute
# permissions and stopping all associated services and processes.
#
stop ()
{
  local _zslaunchd
  local _zsplist

  # Prevent Zscaler from being executed or restarted.
  echo -e "--- Disable: Zscaler app executables"
  sudo chmod -vv a-x "${Z_BIN}"
  sudo chmod -vv a-x "${Z_TNL}"
  sudo chmod -vv a-x "${Z_SRV}"

  echo -e "--- Kill: Zscaler"
  sudo lsof -nP -t -c "/${Z_APPNAME}|${Z_PLUGINS}/i" | xargs -I{} sudo kill -9 {}
  sudo launchctl list | grep -i -e "${Z_APPNAME}" -e "${Z_PLUGINS}" | cut -f 3 | xargs -I{} sudo launchctl bootout "system/{}"
  launchctl list | grep -i -e "${Z_APPNAME}" -e "${Z_PLUGINS}" | cut -f 3 | xargs -I{} sudo launchctl bootout "gui/$(id -u)/{}"
  killall "${Z_APPNAME}" 2>/dev/null || true
}

## Start Zscaler
#
# Enables Zscaler by restoring binary execute permissions and restarts the
# app. A system reboot is recommended for a full service restoration.
#
start ()
{
  # Allow Zscaler to be executed or restarted.
  echo -e "--- Enable: Zscaler app executables"
  sudo chmod -vv a+x "${Z_BIN}"
  sudo chmod -vv a+x "${Z_TNL}"
  sudo chmod -vv a+x "${Z_SRV}"

  echo -e "--- Restart: Zscaler app"
  killall "${Z_APPNAME}" 2>/dev/null || true
  open -a "${Z_APP}" -g
  echo -e "--- IMPORTANT: Reboot your laptop now to complete the Zscaler restart."
}

## Check Zscaler status
#
# Displays active Zscaler network connections, running processes, and the
# current execution status of its core binaries.
#
check ()
{
  echo -e "--- Check: Zscaler open network connections"
  sudo lsof +c0 -Pi -a -c "/${Z_APPNAME}/i"
  echo -e ""

  echo -e "--- Check: Zscaler running processes"
  sudo lsof -nP -t -c "/${Z_APPNAME}|${Z_PLUGINS}/i" | xargs -n1 -I{} ps -p {} -o pid=,command=
  echo -e ""

  echo -e "--- Check: Zscaler app binary"
  local bin
  for bin in "${Z_BIN}" "${Z_TNL}" "${Z_SRV}"; do
    if [[ -x "${bin}" ]]; then
      echo "ENABLED: ${bin}"
    else
      echo "DISABLED: ${bin}"
    fi
  done
  echo -e ""
  echo -e "--- Check: end"
  echo -e ""
}

## Main entry point.
#
# Parse command-line arguments to execute script operations.
#
# @param $1 The operation to perform: "stop", "start", or none for check.
#
# @example
#   main "stop"
#   main "start"
main ()
{
  if [[ "${1:-stop}" = "stop" ]]; then
    check
    stop
  elif [[ "${1:-}" = "start" ]]; then
    start
  else
    check
    echo "Usage: $0 [stop|start]" 1>&2
    exit 1
  fi
}

main "$@"
