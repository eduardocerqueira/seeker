#date: 2025-01-24T17:09:08Z
#url: https://api.github.com/gists/c8b3756cde362543ecd521cadc29b186
#owner: https://api.github.com/users/avianey

#!/bin/bash

# Set locale to ensure decimal point is handled correctly
export LC_NUMERIC="en_US.UTF-8"

# Check if the required tools are installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install ffmpeg and try again."
    exit 1
fi

if ! command -v exiftool &> /dev/null; then
    echo "Error: exiftool is not installed. Please install exiftool and try again."
    exit 1
fi

# Check if the user provided arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <file.MP.jpg> or $0 <*.MP.jpg>"
    exit 1
fi

# Expand input arguments to process files
files=("$@")
total_files="${#files[@]}"
processed_files=0
total_bytes_saved=0

# Function to display the progress bar
show_progress() {
    local current=$1
    local total=$2
    local width=50  # Width of the progress bar
    local percent=$((current * 100 / total))
    local filled=$((percent * width / 100))
    local empty=$((width - filled))
    local bytes_saved_human=$(human_readable_size $total_bytes_saved)
    printf "\r[%-${width}s] %3d%% (%d/%d) - %s | Total Saved: %s" \
        "$(printf "#%.0s" $(seq 1 $filled))" $percent $current $total "$current_file" "$bytes_saved_human"
}

# Function to convert bytes to a human-readable format
human_readable_size() {
    local bytes=$1
    if ((bytes >= 1073741824)); then
        # Convert to GB, rounded to two decimal places
        printf "%.2f GB" "$(echo "scale=2; $bytes / 1073741824" | bc)"
    elif ((bytes >= 1048576)); then
        # Convert to MB, rounded to two decimal places
        printf "%.2f MB" "$(echo "scale=2; $bytes / 1048576" | bc)"
    elif ((bytes >= 1024)); then
        # Convert to kB, rounded to two decimal places
        printf "%.2f kB" "$(echo "scale=2; $bytes / 1024" | bc)"
    else
        # Display bytes directly if less than 1 kB
        printf "%d bytes" "$bytes"
    fi
}

# Process each file
for input_file in "${files[@]}"; do
    processed_files=$((processed_files + 1))
    current_file="$(basename "$input_file")"

    # Check if the file exists
    if [ ! -f "$input_file" ]; then
        echo -e "\nWarning: File '$input_file' not found. Skipping."
        continue
    fi

    # Generate temporary and output filenames
    temp_image="temp_image.jpg"
    output_file="${input_file%.MP.jpg}.jpg"

    # Get the size of the input file
    input_size=$(stat -c%s "$input_file")

    # Extract the first frame as an image (removes MJPEG video stream)
    ffmpeg -i "$input_file" -vf "select=eq(n\,0)" -q:v 2 -frames:v 1 "$temp_image" -y &> /dev/null

    if [ $? -ne 0 ]; then
        echo -e "\nError: Failed to extract the image for $input_file."
        continue
    fi

    # Copy EXIF metadata from the original file to the extracted image
    exiftool -overwrite_original -tagsfromfile "$input_file" "$temp_image" &> /dev/null

    if [ $? -ne 0 ]; then
        echo -e "\nError: Failed to copy EXIF metadata for $input_file."
        rm -f "$temp_image"
        continue
    fi

    # Get the size of the output file
    output_size=$(stat -c%s "$temp_image")
    bytes_saved=$((input_size - output_size))
    total_bytes_saved=$((total_bytes_saved + bytes_saved))

    # Rename the temporary image to the final output file
    mv "$temp_image" "$output_file"

    # Update progress bar
    show_progress "$processed_files" "$total_files"
done

# Print a new line after completing the progress bar
echo -e "\nProcessing complete: $processed_files/$total_files files processed."
echo "Total bytes saved: $(human_readable_size $total_bytes_saved)"
