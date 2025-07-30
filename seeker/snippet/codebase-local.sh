#date: 2025-07-30T17:02:14Z
#url: https://api.github.com/gists/c71b4034477cb9cba05225a811cae4db
#owner: https://api.github.com/users/RezixDev

#!/bin/bash

# Configuration
TARGET_FOLDER="${1:-./src/features/nutrition}"  # Use first argument or default
OUTPUT_FILE="codebase-export.txt"
INCLUDE_EXTENSIONS=("ts" "tsx" "js" "jsx" "css" "html" "json" "md")

# Function to check if file should be included
should_include_file() {
    local file="$1"
    local ext="${file##*.}"
    
    # Check if extension is in our include list
    for include_ext in "${INCLUDE_EXTENSIONS[@]}"; do
        if [[ "$ext" == "$include_ext" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to check if path should be excluded
should_exclude_path() {
    local path="$1"
    
    # Exclude patterns
    if [[ "$path" == *"node_modules"* ]] || \
       [[ "$path" == *".git"* ]] || \
       [[ "$path" == *"dist"* ]] || \
       [[ "$path" == *"build"* ]] || \
       [[ "$path" == *".next"* ]] || \
       [[ "$path" == *".vscode"* ]] || \
       [[ "$path" == *"coverage"* ]] || \
       [[ "$path" == *".nyc_output"* ]] || \
       [[ "$path" == *"temp"* ]] || \
       [[ "$path" == *"tmp"* ]] || \
       [[ "$path" == *".min.js"* ]] || \
       [[ "$path" == *".map"* ]]; then
        return 0
    fi
    return 1
}

# Function to generate tree structure
generate_tree() {
    local target_dir="$1"
    local prefix="$2"
    local is_last="$3"
    
    # Get all items in directory, sorted
    local items=()
    while IFS= read -r -d '' item; do
        local basename=$(basename "$item")
        # Skip excluded paths
        if ! should_exclude_path "$item"; then
            items+=("$item")
        fi
    done < <(find "$target_dir" -maxdepth 1 -mindepth 1 -print0 | sort -z)
    
    local count=${#items[@]}
    local i=0
    
    for item in "${items[@]}"; do
        ((i++))
        local basename=$(basename "$item")
        local is_last_item=false
        [[ $i -eq $count ]] && is_last_item=true
        
        # Choose tree characters
        if $is_last_item; then
            echo "${prefix}└── $basename"
            local new_prefix="${prefix}    "
        else
            echo "${prefix}├── $basename"
            local new_prefix="${prefix}│   "
        fi
        
        # If it's a directory, recurse
        if [[ -d "$item" ]]; then
            generate_tree "$item" "$new_prefix" "$is_last_item"
        fi
    done
}

# Function to get file tree with only included files
generate_filtered_tree() {
    local target_dir="$1"
    local prefix="$2"
    
    # Create temporary structure with only included files
    local dirs=()
    local files=()
    
    # Separate directories and files
    while IFS= read -r -d '' item; do
        if should_exclude_path "$item"; then
            continue
        fi
        
        local basename=$(basename "$item")
        if [[ -d "$item" ]]; then
            # Check if directory contains any included files
            local has_included_files=false
            while IFS= read -r -d '' subfile; do
                if should_include_file "$subfile" && ! should_exclude_path "$subfile"; then
                    has_included_files=true
                    break
                fi
            done < <(find "$item" -type f -print0)
            
            if $has_included_files; then
                dirs+=("$item")
            fi
        elif [[ -f "$item" ]] && should_include_file "$item"; then
            files+=("$item")
        fi
    done < <(find "$target_dir" -maxdepth 1 -mindepth 1 -print0 | sort -z)
    
    # Combine and sort
    local all_items=("${dirs[@]}" "${files[@]}")
    local count=${#all_items[@]}
    local i=0
    
    for item in "${all_items[@]}"; do
        ((i++))
        local basename=$(basename "$item")
        local is_last_item=false
        [[ $i -eq $count ]] && is_last_item=true
        
        # Choose tree characters and add file size for files
        if $is_last_item; then
            if [[ -f "$item" ]]; then
                local size=$(du -h "$item" 2>/dev/null | cut -f1)
                echo "${prefix}└── $basename ${size:+($size)}"
            else
                echo "${prefix}└── $basename/"
            fi
            local new_prefix="${prefix}    "
        else
            if [[ -f "$item" ]]; then
                local size=$(du -h "$item" 2>/dev/null | cut -f1)
                echo "${prefix}├── $basename ${size:+($size)}"
            else
                echo "${prefix}├── $basename/"
            fi
            local new_prefix="${prefix}│   "
        fi
        
        # If it's a directory, recurse
        if [[ -d "$item" ]]; then
            generate_filtered_tree "$item" "$new_prefix"
        fi
    done
}

# Check if target folder exists
if [[ ! -d "$TARGET_FOLDER" ]]; then
    echo "❌ Error: Target folder '$TARGET_FOLDER' does not exist"
    echo "Usage: $0 [folder_path]"
    echo "Example: $0 ./src/components"
    exit 1
fi

echo "🔍 Analyzing folder: $TARGET_FOLDER"
echo "📄 Output file: $OUTPUT_FILE"

# Start building the output
{
    echo "# Codebase Export for Claude"
    echo "Generated on: $(date)"
    echo "Target folder: $TARGET_FOLDER"
    echo ""
    echo "## 📁 Project Structure"
    echo ""
    echo "\`\`\`"
    echo "$(basename "$TARGET_FOLDER")/"
    generate_filtered_tree "$TARGET_FOLDER" ""
    echo "\`\`\`"
    echo ""
    echo "## 📄 File Contents"
    echo ""
} > "$OUTPUT_FILE"

# Process all files in the target folder
file_count=0
declare -A processed_files

while IFS= read -r -d '' file; do
    # Skip if should be excluded
    if should_exclude_path "$file"; then
        continue
    fi
    
    # Skip if not a file we want to include
    if ! should_include_file "$file"; then
        continue
    fi
    
    # Get relative path
    rel_path=$(realpath --relative-to="$TARGET_FOLDER" "$file")
    
    # Skip if already processed (shouldn't happen, but safety check)
    if [[ -n "${processed_files[$rel_path]}" ]]; then
        continue
    fi
    processed_files[$rel_path]=1
    
    # Get file extension for syntax highlighting
    ext="${file##*.}"
    case "$ext" in
        "ts"|"tsx") lang="typescript" ;;
        "js"|"jsx") lang="javascript" ;;
        "css") lang="css" ;;
        "html") lang="html" ;;
        "json") lang="json" ;;
        "md") lang="markdown" ;;
        *) lang="text" ;;
    esac
    
    # Get file info
    local file_size=$(du -h "$file" 2>/dev/null | cut -f1)
    local line_count=$(wc -l < "$file" 2>/dev/null || echo "0")
    
    {
        echo ""
        echo "### 📄 \`$rel_path\`"
        echo ""
        echo "**File Info:** ${file_size:-"Unknown size"} • ${line_count} lines • .$ext file"
        echo ""
        echo "\`\`\`$lang"
        cat "$file"
        echo ""
        echo "\`\`\`"
        echo ""
        echo "---"
    } >> "$OUTPUT_FILE"
    
    ((file_count++))
    echo "📝 Processed: $rel_path (${file_size:-"?"})"
    
done < <(find "$TARGET_FOLDER" -type f -print0 | sort -z)

# Get final file info
if [[ -f "$OUTPUT_FILE" ]]; then
    file_size=$(du -h "$OUTPUT_FILE" | cut -f1)
    line_count=$(wc -l < "$OUTPUT_FILE")
    
    # Add summary to the end of the file
    {
        echo ""
        echo "## 📊 Export Summary"
        echo ""
        echo "- **Target Folder:** \`$TARGET_FOLDER\`"
        echo "- **Files Processed:** $file_count files"
        echo "- **Total Lines:** $line_count lines"
        echo "- **Output Size:** $file_size"
        echo "- **Generated:** $(date)"
        echo "- **Included Extensions:** ${INCLUDE_EXTENSIONS[*]}"
    } >> "$OUTPUT_FILE"
    
    echo ""
    echo "✅ Export completed successfully!"
    echo "📁 Target folder: $TARGET_FOLDER"
    echo "📄 Output file: $OUTPUT_FILE"
    echo "📊 Processed $file_count files"
    echo "📏 Generated $line_count lines ($file_size)"
    echo ""
    echo "🎯 The export includes a visual tree structure and detailed file contents."
else
    echo "❌ Error: Failed to create output file"
    exit 1
fi