#date: 2026-01-27T16:56:31Z
#url: https://api.github.com/gists/58c9d19075ea5dd379279f2134790b54
#owner: https://api.github.com/users/kingdudely

#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

INPUT_FILE=${1:?"Error: No input file provided!"}
OUTPUT_FILE=${2:-"${INPUT_FILE}.js"}

trap "rm -f "$INPUT_FILE".{dsm,bc,ll,config.json,c} && kill 0" EXIT

commandExists() {
	local COMMAND=$1
	command -v "$COMMAND" &>/dev/null
}

ensurePackage() {
	local PACKAGE=$1
	if ! commandExists "$PACKAGE"; then
		echo "Installing missing system dependency: $PACKAGE ..."
		sudo apt update
		sudo apt install -y "$PACKAGE"
	fi
}

ensureRetDec() {
	local TAR_XZ_URL=$1
	local RETDEC_DIRECTORY=$2

	if ! commandExists retdec-decompiler; then
		if [ ! -d "$RETDEC_DIRECTORY" ]; then
			# echo "RetDec is not installed. Installing RetDec..."
	
			TAR_XZ="/tmp/RetDec.tar.xz"
			wget -O "$TAR_XZ" "$TAR_XZ_URL"
	
			rm -rf "$RETDEC_DIRECTORY"
			mkdir -p "$RETDEC_DIRECTORY"
		
			tar -xJf "$TAR_XZ" -C "$RETDEC_DIRECTORY"
			rm -rf "$TAR_XZ"
		fi
	
		sudo chmod -R +x $RETDEC_DIRECTORY/bin
		export PATH=$RETDEC_DIRECTORY/bin:$PATH	
	fi
}

ensureEmscripten() {
	local GIT_URL=$1
	local EMSCRIPTEN_DIRECTORY=$2

	if ! commandExists emcc; then # [ ! -d "./emsdk" ]
		if [ ! -d "$EMSCRIPTEN_DIRECTORY" ]; then
			echo "Emscripten is not installed. Installing Emscripten..."
	
			rm -rf "$EMSCRIPTEN_DIRECTORY"
			mkdir -p "$EMSCRIPTEN_DIRECTORY"
		
			git clone "$GIT_URL" "$EMSCRIPTEN_DIRECTORY"
		fi
	
		sudo chmod +x $EMSCRIPTEN_DIRECTORY/emsdk
		$EMSCRIPTEN_DIRECTORY/emsdk install latest
		$EMSCRIPTEN_DIRECTORY/emsdk activate latest
		source $EMSCRIPTEN_DIRECTORY/emsdk_env.sh
	fi
}

ensurePackage tar
ensurePackage wget
ensurePackage git
ensureRetDec "https://github.com/avast/retdec/releases/download/v5.0/RetDec-v5.0-Linux-Release.tar.xz" "$HOME/.local/retdec"
ensureEmscripten "https://github.com/emscripten-core/emsdk.git" "$HOME/.local/emsdk"

retdec-decompiler \
	"$INPUT_FILE" \
	-f plain \
	-s \
	--backend-no-opts \
	--backend-no-var-renaming \
	--backend-no-compound-operators \
	--backend-no-symbolic-names \
	--no-memory-limit

emcc "${INPUT_FILE}.c" -O3 -s WASM=1 -s SINGLE_FILE=1 -o "$OUTPUT_FILE"

echo "Done! Output: $OUTPUT_FILE"