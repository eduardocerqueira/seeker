#date: 2025-06-18T17:00:10Z
#url: https://api.github.com/gists/02267ab8148bf72397eff8a87fefbb75
#owner: https://api.github.com/users/ccallawa-intel

#!/bin/sh

set -x
set -e

SRC_HASH="$1"
FOSSIL_PATH="$2"
FOSSILIZE_REPLAY_PATH="$HOME/src/Fossilize/build-c03ec90/cli/fossilize-replay"
STATS_DUMP_PATH="/tmp/fossiize-replay/stats.csv"
MESA_SPIRV_DUMP_PATH="/tmp/fossilize-replay/spirv-dump"
SPIRV_CROSS_PATH="/vulkan-sdk/1.4.304.1/x86_64/bin/spirv-cross "

if [ -z "$SRC_HASH" ] || [ -z "$FOSSIL_PATH" ]; then
    echo "Usage: $0 <src_hash> <fossil_path>"
    exit 1
fi

if [ ! -f "$FOSSIL_PATH" ]; then
    echo "Fossil file not found: $FOSSIL_PATH"
    exit 1
fi

if [ ! -x "$FOSSILIZE_REPLAY_PATH" ]; then
    echo "Fossilize replay tool not found or not executable: $FOSSILIZE_REPLAY_PATH"
    exit 1
fi

mkdir -p $(dirname "$STATS_DUMP_PATH")
mkdir -p "$MESA_SPIRV_DUMP_PATH"

$FOSSILIZE_REPLAY_PATH --enable-pipeline-stats $STATS_DUMP_PATH $FOSSIL_PATH
PIPELINE_HASH=$(grep "$SRC_HASH" "$STATS_DUMP_PATH" | cut -d ',' -f3)

if [ -z "$PIPELINE_HASH" ]; then
    echo "Pipeline hash not found for $1"
    exit 1
fi

echo "Pipeline hash for $SRC_HASH: $PIPELINE_HASH"

MESA_SPIRV_DUMP_PATH=$MESA_SPIRV_DUMP_PATH MESA_SHADER_CACHE_DISABLE=1 $FOSSILIZE_REPLAY_PATH --pipeline-hash $PIPELINE_HASH $FOSSIL_PATH

$SPIRV_CROSS_PATH -V "$MESA_SPIRV_DUMP_PATH/spirv-0.spirv" > "0x$SRC_HASH.glsl"
