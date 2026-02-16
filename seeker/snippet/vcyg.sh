#date: 2026-02-16T17:26:53Z
#url: https://api.github.com/gists/1be5ca922c2649ffd4af1b1697088d70
#owner: https://api.github.com/users/op30mmd

#!/bin/bash

# =============================================================================
# vcyg - A wrapper to compile V (vlang) applications strictly inside Cygwin.
#
# It works by transpiling V to Linux-compatible C code, patching the C code 
# to remove Cygwin conflicts (bool, _S macro, ptrace), and compiling with GCC.
# =============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---

# 1. Try to find V in standard locations or use the environment variable $V_PATH
if [ -n "$V_PATH" ]; then
    V_COMPILER="$V_PATH"
elif [ -f "/cygdrive/c/v/v.exe" ]; then
    V_COMPILER="/cygdrive/c/v/v.exe"
elif [ -f "$HOME/v/v.exe" ]; then
    V_COMPILER="$HOME/v/v.exe"
else
    echo -e "\033[31mError: V compiler (v.exe) not found.\033[0m"
    echo "Please set the V_PATH variable or install V in C:\v"
    exit 1
fi

# Colors for nice output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Argument Parsing ---

INPUT_ARGS=("$@")
V_FLAGS=("-os" "linux" "-gc" "none" "-d" "no_backtrace") # Force Linux target, disable GC/Backtrace
USER_FLAGS=()
TARGET="." # Default to current directory

# Separate user flags from the target file/directory
for arg in "${INPUT_ARGS[@]}"; do
    if [[ "$arg" == -* ]]; then
        USER_FLAGS+=("$arg")
    else
        TARGET="$arg"
    fi
done

# Determine Output Name
# If the target is a directory, name the exe after the directory.
# If it's a file, name it after the file.
if [ -d "$TARGET" ] || [ "$TARGET" == "." ]; then
    FULL_PATH=$(realpath "$TARGET")
    BASE_NAME=$(basename "$FULL_PATH")
else
    BASE_NAME=$(basename "$TARGET" .v)
fi

# Define temporary filenames
C_FILE="${BASE_NAME}.tmp.c"
EXE_FILE="${BASE_NAME}.exe"

# Handle custom output name if user passed -o
for ((i=0; i<${#USER_FLAGS[@]}; i++)); do
    if [[ "${USER_FLAGS[$i]}" == "-o" ]]; then
        EXE_FILE="${USER_FLAGS[$i+1]}"
        # Remove -o and the filename from flags passed to V transpiler
        unset 'USER_FLAGS[$i]'
        unset 'USER_FLAGS[$i+1]'
        break
    fi
done

# --- Step 1: Transpilation ---

echo -e "${BLUE}[vcyg] Transpiling '${TARGET}' to C (Target: Linux)...${NC}"

# We invoke V to generate the C file, but we do NOT let it compile (it would fail).
# We pass user flags (like -prod) here so the C code is optimized if requested.
"$V_COMPILER" "${V_FLAGS[@]}" "${USER_FLAGS[@]}" -o "$C_FILE" "$TARGET"

if [ ! -f "$C_FILE" ]; then
    echo -e "${RED}[vcyg] Error: Failed to generate C file.${NC}"
    exit 1
fi

# --- Step 2: Patching ---

echo -e "${BLUE}[vcyg] Patching C code for Cygwin compatibility...${NC}"

# This sed command performs the "surgery" needed to make V's C code
# compatible with Cygwin's system headers.
sed -i \
    -e '/#error Cygwin is not supported/d' \
    -e '/#error VERROR_MESSAGE Header file <sys\/ptrace.h>/d' \
    -e '/#error VERROR_MESSAGE Header file <gc.h>/d' \
    -e 's/typedef u8 bool;//' \
    -e 's/_S(/_VSTR(/g' \
    -e 's/#define _S(/#define _VSTR(/' \
    "$C_FILE"

# --- Step 3: Compilation ---

echo -e "${BLUE}[vcyg] Compiling '${EXE_FILE}' with GCC...${NC}"

# Compile using GCC.
# -D__linux__    : Tricks V code into using POSIX paths/logic.
# -Dgettid()=0   : Stubs out the gettid syscall (missing in Cygwin).
# -w             : Suppress warnings (V generated C often has many benign warnings).
gcc -D__linux__ -D"gettid()=0" -w "$C_FILE" -o "$EXE_FILE"

# --- Step 4: Cleanup ---

if [ -f "$EXE_FILE" ]; then
    echo -e "${GREEN}[vcyg] Build successful: ./${EXE_FILE}${NC}"
    rm -f "$C_FILE" # Remove the temp C file on success
else
    echo -e "${RED}[vcyg] GCC compilation failed. Keeping $C_FILE for inspection.${NC}"
    exit 1
fi