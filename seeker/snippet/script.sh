#date: 2023-04-27T16:42:15Z
#url: https://api.github.com/gists/381b5c732c9fb51c0714fac9124f2159
#owner: https://api.github.com/users/MaheshRavishankar

#!/bin/bash

set -e
set -x

HOME_DIR="${HOME_DIR:-$HOME}"
IREE_BUILD_DIR="${IREE_BUILD_DIR:-$HOME_DIR/iree/build_relwithdebinfo}"
LLVM_BIN_DIR="${LLVM_BIN_DIR:-$IREE_BUILD_DIR/llvm-project/bin}"
CLANG="${CLANG:-$LLVM_BIN_DIR/clang}"
LLVM_LINK="${LLVM_LINK:-$LLVM_BIN_DIR/llvm-link}"
LLVM_DIS="${LLVM_DIS:-$LLVM_BIN_DIR/llvm-dis}"
IREE_DIR="${IREE_DIR:-$HOME/iree/iree}"
RUNTIME_DIR="${RUNTIME_DIR:-$IREE_DIR/runtime/src/}"
BUILTIN_DIR="${BUILTIN_DIR:-$RUNTIME_DIR/iree/builtins/ukernel}"
RUNTIME_BIN_DIR="${RUNTIME_BIN_DIR:-${IREE_BUILD_DIR}/runtime/src}"
LIBDEVICE_FLAGS="-std=c17 -nostdinc -ffreestanding -O3 -fno-ident -fdiscard-value-names -c -emit-llvm -DIREE_DEVICE_STANDALONE=1"

files=(
  "mmt4d.c"
  "mmt4d_tile.c"
  "arch/x86_64/mmt4d_x86_64.c"
)

bc_files=()
for file in "${files[@]}"
do
  file_name=`echo ${file} | awk -F/ '{print $NF}'`
  bc_file=${file_name}.bc
  ${CLANG} ${LIBDEVICE_FLAGS} -mavx2 -mfma ${BUILTIN_DIR}/${file} -I${RUNTIME_DIR} -I${RUNTIME_BIN_DIR} -o ${bc_file}
  bc_files+=("${bc_file}")
done

${LLVM_LINK} "${bc_files[@]}" -o mmt4d_link.bc
${LLVM_DIS} mmt4d_link.bc -o mmt4d_link.ll
