#date: 2025-06-10T16:54:44Z
#url: https://api.github.com/gists/92d0e0aff888e33abb968241957f45d4
#owner: https://api.github.com/users/bk-rubrik

#!/bin/bash

# --- Configuration ---
DISK_MOUNT_PATH="/tmp/lz4_data" # Make sure your 400GB disk is mounted here
OUTPUT_FILE="${DISK_MOUNT_PATH}/pattern_data.bin"
TOTAL_DISK_SIZE_GB=400
FILL_PERCENTAGE=96 # Fill 96% of the disk
BLOCK_SIZE_KB=512  # Size of the repeating pattern block (our 50% random/50% zero block)

# Calculate sizes in bytes
TOTAL_DISK_SIZE_BYTES=$(( TOTAL_DISK_SIZE_GB * 1024 * 1024 * 1024 ))
TARGET_FILL_SIZE_BYTES=$(( TOTAL_DISK_SIZE_BYTES * FILL_PERCENTAGE / 100 ))
BLOCK_SIZE_BYTES=$(( BLOCK_SIZE_KB * 1024 ))

# Calculate sizes for random and zero parts within the block
# 50% random, 50% zeros
RANDOM_PART_SIZE=$(( BLOCK_SIZE_BYTES / 2 ))
ZERO_PART_SIZE=$(( BLOCK_SIZE_BYTES - RANDOM_PART_SIZE ))

# Calculate total number of 512KB blocks needed
TOTAL_BLOCKS_NEEDED=$(( TARGET_FILL_SIZE_BYTES / BLOCK_SIZE_BYTES ))
if [ $(( TARGET_FILL_SIZE_BYTES % BLOCK_SIZE_BYTES )) -ne 0 ]; then
    TOTAL_BLOCKS_NEEDED=$(( TOTAL_BLOCKS_NEEDED + 1 )) # Add one for the partial block if any
fi

echo "--- Data Generation Script ---"
echo "Target disk size: ${TOTAL_DISK_SIZE_GB} GB"
echo "Fill percentage: ${FILL_PERCENTAGE}%"
echo "Target fill size: $(( TARGET_FILL_SIZE_BYTES / (1024*1024*1024) )) GB"
echo "Repeating block size: ${BLOCK_SIZE_KB} KB"
echo "Output file: ${OUTPUT_FILE}"
echo "Total blocks to write: ${TOTAL_BLOCKS_NEEDED}"

# --- Create the 512KB pattern block ---
# Create a temporary file to hold the single repeating block
TEMP_PATTERN_BLOCK="/tmp/single_pattern_block.tmp"
echo "Creating the 512KB repeating pattern block in ${TEMP_PATTERN_BLOCK}..."

# Generate 50% random data
dd if=/dev/urandom of="${TEMP_PATTERN_BLOCK}" bs=1 count=${RANDOM_PART_SIZE} status=none 2>/dev/null

# Generate 50% zero bytes and append it
dd if=/dev/zero of="${TEMP_PATTERN_BLOCK}" bs=1 count=${ZERO_PART_SIZE} oflag=append conv=notrunc status=none 2>/dev/null

# Verify the size of the generated pattern block
GENERATED_PATTERN_SIZE=$(stat -c%s "${TEMP_PATTERN_BLOCK}")
echo "Generated pattern block size: $(( GENERATED_PATTERN_SIZE / 1024 )) KB (Expected: ${BLOCK_SIZE_KB} KB)"

if [ "$GENERATED_PATTERN_SIZE" -ne "$BLOCK_SIZE_BYTES" ]; then
    echo "Error: Generated pattern block size is incorrect. Exiting."
    rm -f "${TEMP_PATTERN_BLOCK}"
    exit 1
fi

# --- Repeat the block to fill the disk ---
echo "Writing repeating blocks to ${OUTPUT_FILE}..."
echo "This may take a while for 384GB of data."

BYTES_WRITTEN=0
BLOCKS_WRITTEN=0
START_TIME=$(date +%s)

# Create an empty file first, or `cat` will append if it exists from previous runs
truncate -s 0 "${OUTPUT_FILE}"

# Open the output file for appending once, and keep it open
# This is much faster than repeatedly opening/closing/seeking with dd
exec 3>> "${OUTPUT_FILE}"

while [ "$BYTES_WRITTEN" -lt "$TARGET_FILL_SIZE_BYTES" ]; do
    # Calculate how much more to write to reach target
    REMAINING_BYTES=$(( TARGET_FILL_SIZE_BYTES - BYTES_WRITTEN ))

    # Determine how many full blocks to write in this pass (e.g., 1000 blocks at a time for efficiency)
    # Using a buffer of 1000 blocks (approx 512MB) per cat operation
    BATCH_SIZE=$(( 1000 * BLOCK_SIZE_BYTES ))
    CURRENT_WRITE_SIZE=$(( REMAINING_BYTES < BATCH_SIZE ? REMAINING_BYTES : BATCH_SIZE ))

    # Calculate how many copies of the pattern block are needed for this write
    COPIES_IN_BATCH=$(( CURRENT_WRITE_SIZE / BLOCK_SIZE_BYTES ))
    if [ $(( CURRENT_WRITE_SIZE % BLOCK_SIZE_BYTES )) -ne 0 ]; then
        COPIES_IN_BATCH=$(( COPIES_IN_BATCH + 1 )) # For partial blocks
    fi

    # Write the pattern block multiple times.
    # We use `head -c` to ensure we don't write more than needed for the last partial chunk.
    # `cat` is generally faster for sequential writes than `dd` for simple copying.
    # The `>>` operator appends to the file.
    for ((i=0; i<COPIES_IN_BATCH; i++)); do
        if [ "$BYTES_WRITTEN" -ge "$TARGET_FILL_SIZE_BYTES" ]; then
            break # Stop if target reached
        fi

        CURRENT_BLOCK_TO_WRITE_SIZE=$(( (BYTES_WRITTEN + BLOCK_SIZE_BYTES) <= TARGET_FILL_SIZE_BYTES ? BLOCK_SIZE_BYTES : (TARGET_FILL_SIZE_BYTES - BYTES_WRITTEN) ))

        if [ "$CURRENT_BLOCK_TO_WRITE_SIZE" -gt 0 ]; then
             cat "${TEMP_PATTERN_BLOCK}" | head -c "$CURRENT_BLOCK_TO_WRITE_SIZE" >&3
             BYTES_WRITTEN=$(( BYTES_WRITTEN + CURRENT_BLOCK_TO_WRITE_SIZE ))
             BLOCKS_WRITTEN=$(( BLOCKS_WRITTEN + 1 ))
        fi
    done


    # Update progress
    ELAPSED_TIME=$(( $(date +%s) - START_TIME ))
    if [ "$ELAPSED_TIME" -gt 0 ]; then
        SPEED_MBPS=$(echo "scale=2; ${BYTES_WRITTEN} / (1024 * 1024) / ${ELAPSED_TIME}" | bc)
        printf "\r  Written $(( BYTES_WRITTEN / (1024*1024*1024) )) GB / $(( TARGET_FILL_SIZE_BYTES / (1024*1024*1024) )) GB (%s%%) at %s MB/s" \
               $(( (BYTES_WRITTEN * 100) / TARGET_FILL_SIZE_BYTES )) "$SPEED_MBPS"
    fi
done

# Close the file descriptor
exec 3>&-

echo -e "\nData generation complete."

# --- Clean up ---
echo "Cleaning up temporary pattern block file..."
rm -f "${TEMP_PATTERN_BLOCK}"

FINAL_SIZE=$(stat -c%s "${OUTPUT_FILE}")
echo "Final file size on disk: $(( FINAL_SIZE / (1024*1024*1024) )) GB"
echo "Script finished."