#date: 2022-07-29T17:06:53Z
#url: https://api.github.com/gists/a71eba185c900f47b82007470dc377d1
#owner: https://api.github.com/users/asherf

#!/usr/bin/env bash

# ***************************** TOOLCHAIN NOTE *************************************************************
# This script is based on the pants script (github.com/pantsbuild/setup)
# However, it will not actually bootstrap pants, it it expects pants to already be bootstrapped and will
# fail if it doesn't.
# ***********************************************************************************************************


set -eou pipefail

PYTHON_BIN_NAME="python3"

PANTS_BIN_NAME="${PANTS_BIN_NAME:-$0}"
PANTS_BOOTSTRAP="${HOME}/.cache/pants/setup/bootstrap-$(uname -s)-$(uname -m)"

COLOR_RED="\x1b[31m"
COLOR_GREEN="\x1b[32m"
COLOR_RESET="\x1b[0m"

function log() {
  echo -e "$@" 1>&2
}

function die() {
  (($# > 0)) && log "${COLOR_RED}$*${COLOR_RESET}"
  exit 1
}

function green() {
  (($# > 0)) && log "${COLOR_GREEN}$*${COLOR_RESET}"
}

function get_exe_path_or_die {
  local exe="$1"
  if ! command -v "${exe}"; then
    die "Could not find ${exe}. Please ensure ${exe} is on your PATH."
  fi
}


function get_python_major_minor_version {
  local python_exe="$1"
  "$python_exe" <<EOF
import sys
major_minor_version = ''.join(str(version_num) for version_num in sys.version_info[0:2])
print(major_minor_version)
EOF
}

function set_supported_python_versions {
  # 3.9 only
  supported_python_versions_decimal=('3.9')
  supported_python_versions_int=('39')
  supported_message='3.9'
}

function check_python_exe_compatible_version {
  local python_exe="$1"
  local major_minor_version
  major_minor_version="$(get_python_major_minor_version "${python_exe}")"
  for valid_version in "${supported_python_versions_int[@]}"; do
    if [[ "${major_minor_version}" == "${valid_version}" ]]; then
      echo "${python_exe}" && return 0
    fi
  done
}


function determine_python_exe {
  local pants_version="$1"
  set_supported_python_versions "${pants_version}"
  local requirement_str="For \`pants_version = \"${pants_version}\"\`, Pants requires Python ${supported_message} to run."
  local python_exe
  python_exe="$(get_exe_path_or_die "${PYTHON_BIN_NAME}")" || exit 1
  if [[ -z "$(check_python_exe_compatible_version "${python_exe}")" ]]; then
    die "Invalid Python interpreter version for ${python_exe}. ${requirement_str}"
  fi
  echo "${python_exe}"
}


function bootstrap_pants {
  local pants_version="$1"
  local python="$2"
  local pants_requirement="pantsbuild.pants==${pants_version}"
  local python_major_minor_version
  python_major_minor_version="$(get_python_major_minor_version "${python}")"
  local target_folder_name="${pants_version}_py${python_major_minor_version}"
  local bootstrapped="${PANTS_BOOTSTRAP}/${target_folder_name}"

  if [[ ! -d "${bootstrapped}" ]]; then
      echo "PANTS NOT FOUND IN ${bootstrapped} !!"
      exit 1
  fi
  echo "${bootstrapped}"
}

# Ensure we operate from the context of the ./pants buildroot.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
pants_version="${EXPECTED_PANTS_VERSION}"
python="$(determine_python_exe "${pants_version}")"
pants_dir="$(bootstrap_pants "${pants_version}" "${python}")" || exit 1
pants_python="${pants_dir}/bin/python"
pants_binary="${pants_dir}/bin/pants"

# shellcheck disable=SC2086
exec "${pants_python}" "${pants_binary}" --pants-bin-name="${PANTS_BIN_NAME}" --pants-version=${pants_version} "$@"
