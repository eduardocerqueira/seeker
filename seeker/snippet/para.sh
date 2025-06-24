#date: 2025-06-24T17:14:52Z
#url: https://api.github.com/gists/3a80d6fff551806dccd5fa9be0008b7f
#owner: https://api.github.com/users/johnmeade

#!/bin/bash
# MIT License / John Meade

print_usage() {
  echo "Parallel file sync / remove."
  echo
  echo "Installation:"
  echo "  gcc --version # eg 'sudo apt install build-essential'"
  echo "  sudo chmod 755 para"
  echo "  sudo mv para /usr/local/bin"
  echo "  sudo para # (first time setup)"
  echo
  echo "Usage:"
  echo "    para {cp|rm} [-L] [-n NUM_WORKERS] src [dst]"
  echo
  echo "Args:"
  echo "    -L: follow symlinks"
  echo
  echo "Example 1: Copy, following symlinks"
  echo "    para cp -L /data/foo /data/backup"
  echo
  echo "Example 2: Remove all files in a folder"
  echo "    para rm /data/bar"
}

if [ "$1" = "-h" ] || [ "$1" == "--help" ] || [ "$1" == "help" ]
then
  print_usage
  exit 0
fi

#
#  Functions
#

pretty_print_duration() {
  local start_ns=$1
  local end_ns=$2
  local elapsed_ns=$((end_ns - start_ns))

  local ms=$(( (elapsed_ns / 1000000) % 1000 ))
  local s=$(( (elapsed_ns / 1000000000) % 60 ))
  local m=$(( (elapsed_ns / 60000000000) % 60 ))
  local h=$(( elapsed_ns / 3600000000000 ))

  [ $h -gt 0 ] && out+=" ${h}h"
  [ $m -gt 0 ] && out+=" $(printf "%02d" $m)m"
  [ $s -gt 0 ] && out+=" $(printf "%02d" $s)s"
  out+=" $(printf "%03d" $ms)ms"

  echo "$out"
}

# Usage:
#   build_c_helper "progress_counter" "$PROGRESS_C_SRC"
build_c_helper() {
  if ! ( gcc --version > /dev/null )
  then
    echo "gcc not found, please install build-essential or similar"
    exit 1
  fi

  local name="$1"       # e.g. "progress_counter"
  local src="$2"        # multiline string containing C code
  local helper="/usr/local/bin/$name"
  local helperc="/tmp/${name}.$$.$RANDOM.c"
  local helpertmp="/tmp/${name}.$$.$RANDOM"

  if [[ ! -x "$helper" ]]; then
    echo "Building '$name' (sudo required)"
    sudo true

    # Write the C source to a temp file
    echo "$src" > "$helperc"

    set -e
    gcc -O2 -o "$helpertmp" "$helperc"
    sudo mv "$helpertmp" "$helper"
    rm -f "$helperc"
    sudo chmod 755 "$helper"
    sudo chown root:root "$helper"
    set +e

    echo "Installed $helper"
  fi
}

# Portable alternative to `pv` to show progress while running.
# (inline C to eliminate bash overhead)
PROGRESS_C_SRC='
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  int count = 0;
  struct timespec start, now;
  clock_gettime(CLOCK_MONOTONIC, &start);

  int c;
  while ((c = getchar()) != EOF) {
    putchar(c);
    if (c == '\0') {
      count++;
      if (count % 1000 == 0) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        long elapsed = now.tv_sec - start.tv_sec;
        long hours = elapsed / 3600;
        long minutes = (elapsed % 3600) / 60;
        long seconds = elapsed % 60;

        fprintf(stderr, "\r[ Processed %d files in %ldh %02ldm %02lds ]",
                count, hours, minutes, seconds);
        fflush(stderr);
      }
    }
  }
  fprintf(stderr, "\n");
  return 0;
}
'
build_c_helper "progress_counter" "$PROGRESS_C_SRC"

# Compare 2 files and guess if they are different by filesize
# (inline C to eliminate bash overhead)
CMP_CP_C_SRC='
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// Check file size via stat()
static int get_file_size(const char *path, off_t *size) {
    struct stat st;
    if (stat(path, &st) == 0) {
        *size = st.st_size;
        return 0;
    }
    return -1;
}

// Perform buffered copy
static int copy_file(const char *src_path, const char *dst_path) {
    FILE *src = fopen(src_path, "rb");
    if (!src) return 1;

    FILE *dst = fopen(dst_path, "wb");
    if (!dst) {
        fclose(src);
        return 1;
    }

    char buf[16384];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), src)) > 0) {
        if (fwrite(buf, 1, n, dst) != n) {
            fclose(src);
            fclose(dst);
            return 1;
        }
    }

    fclose(src);
    fclose(dst);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s src_fp dst_fp\n", argv[0]);
        return 1;
    }

    const char *src_fp = argv[1];
    const char *dst_fp = argv[2];

    off_t src_size = 0, dst_size = -1;
    if (get_file_size(src_fp, &src_size) != 0) {
        fprintf(stderr, "Could not stat source file: %s\n", src_fp);
        return 1;
    }

    get_file_size(dst_fp, &dst_size);  // OK if this fails (means dst doesnt exist)

    if (src_size != dst_size) {
        return copy_file(src_fp, dst_fp);
    }

    // File already exists and is the same size — do nothing
    return 0;
}
'
build_c_helper "compare_and_copy" "$CMP_CP_C_SRC"

#
#  Args
#

# Check required args
if [ "$#" -lt 2 ]; then
  print_usage
  exit 1
fi

# Parse subcommand first
mode="$1"
shift
if [[ "$mode" != "cp" && "$mode" != "rm" ]]; then
  echo "Error: first arg must be 'cp' or 'rm'"
  print_usage
  exit 1
fi

# Defaults
follow_symlinks=""
num_workers=32
positional=()

# Manually parse args to allow flags before/after positional
while [[ $# -gt 0 ]]; do
  case "$1" in
    -L)
      follow_symlinks="-L"
      shift
      ;;
    -n)
      shift
      num_workers="$1"
      shift
      ;;
    -*)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

# Assign positional args
src="${positional[0]}"
dst="${positional[1]}"

# Validate positional args
if [[ "$mode" == "cp" && ( -z "$src" || -z "$dst" ) ]]; then
  echo "cp mode requires: src and dst"
  print_usage
  exit 1
elif [[ "$mode" == "rm" && -z "$src" ]]; then
  echo "rm mode requires: src"
  print_usage
  exit 1
fi

# Ensure src is a directory
if [ ! -d "$src" ]; then
  echo "Source must be a directory: $src"
  exit 1
fi

#
#  Main
#

t0=$(date +%s%N)

case "$mode" in
  cp)
    if [ -z "$dst" ]; then
      echo "Destination required for cp mode"
      exit 1
    fi

    echo "[ Copying from $src to $dst with $num_workers workers ]"
    pushd "$src" > /dev/null

    # Create dir structure
    find . $follow_symlinks -type d -print0 | xargs -0 -I{} mkdir -p "$dst/{}"

    # Copy only files that are new or different in size
    find . $follow_symlinks -type f -print0 | \
      progress_counter | \
      xargs -0 -P "$num_workers" -I{} compare_and_copy "./{}" "$dst/{}"

    popd > /dev/null
    echo "[ Done Copying ]"
    ;;

  rm)
    echo "[ Deleting files in $src with $num_workers workers ]"

    find "$src" $follow_symlinks -type f -print0 | \
      progress_counter | \
      xargs -0 -P "$num_workers" -I{} rm -f "{}"

    echo "[ Done Deleting Files ]"
    ;;

  *)
    echo "Unsupported mode: $mode (must be cp or rm)"
    exit 1
    ;;

esac

t1=$(date +%s%N)
echo "[ Took $(pretty_print_duration "$t0" "$t1") ]"
