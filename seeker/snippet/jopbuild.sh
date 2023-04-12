#date: 2023-04-12T16:46:25Z
#url: https://api.github.com/gists/92f20fd426d01011b68c384157bb5378
#owner: https://api.github.com/users/disaac

#!/usr/bin/env bash
# Modified from https://github.com/laurent22/joplin/issues/6052#issuecomment-1356864011
# Tested on latest dev branch as of 20213-04-12
function initVars() {
  REPO="${JOP_REPO:-"laurent22/joplin"}"
  repoUrl="https://github.com/${REPO}.git"
  srcBaseDir="joplin"
  appDesktopPath="packages/app-desktop"
  packageJsonPath="${appDesktopPath}/package.json"
  dmgPath="node_modules/dmg-builder/out/dmg.js"
  builtAppPath="dist/mac-arm64/Joplin.app"
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd || exit 255)"
  SCRIPT_SRC="${BASH_SOURCE[0]}"
  R_SCRIPT_DIR="$(realpath "${SCRIPT_DIR}")"
  R_SCRIPT_SRC="$(realpath "${SCRIPT_SRC}")"
  SCRIPT_NAME="$(basename "${R_SCRIPT_SRC}")"
  dlBranch=""
  preRelease=0
  tagRelease=""
  latestRelease=""
  debugBuild=0
  outDir="${HOME}/Downloads"
}
function checkDeps() {
  local binDeps
  binDeps=(
    sed
    jq
    python2
    jo
    pod
    vips
    yarn
    node
    npx
    git
    curl
    security
  )
  for bin in "${binDeps[@]}"; do
    command -v "${bin}" &>/dev/null || {
      echo >&2 "Required binary ${bin} not found exiting. binDeps: ${binDeps[*]}"
      exit 254
    }
  done
  # Default case for Linux sed, just use "-i"
  sedi=(-i)
  case "$(uname)" in
    # For macOS, use two parameters if its not the gnu version which has a --version
    Darwin*) if ! sed --version >/dev/null 2>&1; then sedi=(-i ""); fi;;
  esac
  # Check for valid codesigning identity if not disable it in electron-builder
  if ! security find-identity -v -p codesigning | grep -Eq "[0-9]\)"; then
    export CSC_IDENTITY_AUTO_DISCOVERY=false
    unset CSC_LINK
    unset CSC_KEY_PASSWORD
    unset CSC_NAME
    unset CSC_KEYCHAIN
  fi
}
function help() {
  echo "Builds Joplin [version] in ARM64, e.g.:"
  echo "Usage: $SCRIPT_NAME [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -h, --help                Show this help message"
  echo "  -p, --pre                 Get the latest pre-release"
  echo '  -l, --latest              Get the latest release'
  echo '  -t, --tag    "tag"        Get release with tag "tag"'
  echo '  -o, --outdir "path"       Set output directory to "path"'
  echo '  -d, --debug               Run script with debug on'
  echo
  echo "Before running for the first time, you should install dependancies by running"
  echo "brew install node yarn cocoapods vips jq jo"
  exit 0
}
function parseArgs() {
  [[ $# -gt 0 ]] || help
  # Parse command line arguments
  while [[ $# -gt 0 ]]; do
    arg="$1"
    case $arg in
      -h | --help)
        help
        ;;
      -p | --pre)
        preRelease=1
        shift
        ;;
      -l | --latest)
        latestRelease=1
        shift
        ;;
      -t | --tag)
        tagRelease="$2"
        shift
        shift
        ;;
      -o | --outdir)
        outDir="$2"
        shift
        shift
        ;;
      -d | --debug)
        debugBuild=1
        shift
        ;;
      *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
  done
}
function setDebug() {
  [[ "${debugBuild}" == 1 ]] && {
    set -x
    echo "Debug flag set so enabled tracing"
  }
}
function gretRelease() {
  if [[ ${preRelease} == "1" ]] && [[ ${latestRelease} != "1" ]]; then
    dlBranch="$(jq -r 'map(select(.prerelease)) | first | .tag_name // "Not found"' <<<"$(curl -s "https://api.github.com/repos/${REPO}/releases")")"
  elif [[ ${latestRelease} == "1" ]] && [[ ${preRelease} != "1" ]]; then
    dlBranch="$(curl -s "https://api.github.com/repos/${REPO}/releases/latest" | jq -r '.tag_name')"
  elif [[ ${preRelease} != "1" ]] && [[ ${latestRelease} != "1" ]] && [[ ${tagRelease} != "" ]]; then
    dlBranch="$(jq -r 'map(select(.tag_name=="'"${tagRelease}"'")) | first | .tag_name // "Not found"' <<<"$(curl -s "https://api.github.com/repos/${REPO}/releases")")"
    [[ -n "${dlBranch}" ]] || {
      echo >&2 "Tag not found please provide a valid tag or use -p or -t instead. Exiting"
      exit 252
    }
  else
    echo "prerelease latest or tag must be provided prerelease:${preRelease} latest:${latestRelease} tag:${tagRelease} dlBranch: ${dlBranch}"
    exit 252
  fi
}

function cloneRepo() {
  local ver="${1}"
  echo "Cloning the requested version ${ver} of Joplin from Github to ${outDir}"
  if [[ ! -d ${outDir} ]]; then
    echo "${outDir} doesn't exist creating before cloning"
    mkdir -p "${outDir}"
  fi
  cd "${outDir}" || {
    echo >&2 "unable to enter ${outDir} exiting as a result."
    exit 254
  }
  git clone --depth 1 --branch "${ver}" "${repoUrl}" || {
    echo >&2 "unable to clone ${repoUrl} exiting as a result."
    exit 253
  }

}
function cleanup {
  local npxRet
  npxRet=$1
  # Remove the files downloaded from Github if all went well
  if [[ $npxRet -eq 0 ]] && [[ "${debugBuild}" != "1" ]]; then
    echo "Remove the files in ${outDir}/${srcBaseDir} downloaded from Github since all went well"
    rm -rf "${outDir:?}/${srcBaseDir:?}" || {
      echo >&2 "unable to remove ${outDir}/${srcBaseDir} exiting as a result."
      exit 252
    }
  else
    echo "Debug flag on debugBuild: $debugBuild or build error: ${npxRet} so not deleting downloaded source."
    echo "To remove manually execute:"
    echo "rm -rf ${outDir}/${srcBaseDir}"
    echo ""
  fi
  # Make a beep to signal it's over
  printf '\a'
}
function buildJoplin() {
  local npxRet
  # Modifying the target for the build (Apple Silicon instead of Intel)
  cd "${srcBaseDir}" || {
    echo >&2 "unable to enter ${srcBaseDir} exiting as a result."
    exit 254
  }
  TMP=$(mktemp)
  VALUE=$(jo target=default "arch[]=arm64")
  jq ".build.mac.target=$VALUE" "${packageJsonPath}" >"$TMP"
  mv "$TMP" "${packageJsonPath}" || {
    echo >&2 "unable to mv $TMP to $packageJsonPath exiting as a result."
    exit 253
  }

  # Downloading and building dependancies
  echo "Downloading and building dependancies"
  yarn install
  cd "${appDesktopPath}" || {
    echo >&2 "unable to enter ${appDesktopPath} exiting as a result."
    exit 254
  }
  # On earlier builds python 2 was required and hardcoded. Check if
  # It needs to be changed in the build file and change.
  # Not required after this merge https://github.com/electron-userland/electron-builder/pull/6617
  if grep -q '/usr/bin/python' "${dmgPath}"; then
    echo "Changing hardcoded python path in ${dmgPath} to python2 path"
    sed "${sedi[@]}" -e "s#/usr/bin/python#$(which python2)#" "${dmgPath}"
  fi
  # Let's finally build Joplin!
  echo "Let's finally build Joplin!"
  npx electron-builder
  npxRet=$?

  # Joplin.app will be copied to the root of the outDir folder
  echo "Copy Joplin.app to the root of ${outDir} folder"
  cp -R "${builtAppPath}" "${outDir}"
  cd "${outDir}" || {
    echo >&2 "unable to enter ${outDir} exiting as a result."
    exit 254
  }
  cleanup $npxRet
}

function main() {
  initVars
  checkDeps
  parseArgs "$@"
  setDebug
  gretRelease
  cloneRepo "${dlBranch}"
  buildJoplin
}
main "$@"
