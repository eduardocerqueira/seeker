#date: 2025-05-13T17:13:15Z
#url: https://api.github.com/gists/287ad95138c53e06a219690f8bcfb966
#owner: https://api.github.com/users/akunzai

#!/usr/bin/env bash
set -euo pipefail

# --- Default Settings ---
VERSION_DEFAULT="main"
README_API_URL="https://dash.readme.com/api/v1"
SCRIPT_DIR=$(dirname "$0")
EXPORT_HIDDEN_DOCS_DEFAULT="false" # Default: Do not export hidden documents

# --- Function: Output error message and exit ---
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# --- Function: Output warning message ---
warning_msg() {
    echo "Warning: $1" >&2
}

# --- Read .env configuration file (if it exists) ---
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Reading configuration from $SCRIPT_DIR/.env..."
    set -o allexport
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/.env"
    set +o allexport
    echo ".env file loaded."
fi

# --- Get ReadMe API Key ---
README_API_KEY="${README_API_KEY:-}"
while [[ -z "${README_API_KEY}" ]]; do
    read -r -p "Please enter your ReadMe API Key: " README_API_KEY
done

# --- Get Project Version ---
if [[ -z "${PROJECT_VERSION:-}" ]]; then
    read -r -p "Enter the project version to export (default: $VERSION_DEFAULT): " PROJECT_VERSION_INPUT
    PROJECT_VERSION="${PROJECT_VERSION_INPUT:-$VERSION_DEFAULT}"
else
    echo "Reading project version from environment variable: $PROJECT_VERSION"
fi

# --- Determine whether to export hidden documents ---
FINAL_EXPORT_HIDDEN_DOCS="${EXPORT_HIDDEN_DOCS:-$EXPORT_HIDDEN_DOCS_DEFAULT}"
echo "Preparing to export project version: $PROJECT_VERSION"
if [ "$FINAL_EXPORT_HIDDEN_DOCS" == "true" ]; then
    echo "Hidden documents will be exported."
fi

# --- Set Output Directory ---
OUTPUT_DIR_BASE="$SCRIPT_DIR/exported"
OUTPUT_DIR="$OUTPUT_DIR_BASE/$PROJECT_VERSION"
mkdir -p "$OUTPUT_DIR"
echo "All documents will be exported to: $OUTPUT_DIR"
echo "Images will be stored in the 'images' subdirectory of each category."
echo ""

# --- Get Category List ---
echo "Fetching category list..."
categories_json=$(curl -s -H "x-readme-version: $PROJECT_VERSION" -u "$README_API_KEY:" "$README_API_URL/categories")

if [ -z "$categories_json" ]; then
    error_exit "Could not fetch category list, or API response was empty. Check API key, version, and network connection."
fi
if echo "$categories_json" | jq -e 'if type == "object" then .error else null end' | grep -q -v null; then
    error_exit "API response for category list contains an error: $(echo "$categories_json" | jq '.')"
fi

num_categories=$(echo "$categories_json" | jq '. | length')
if [ "$num_categories" -eq 0 ]; then
    warning_msg "No categories found in version '$PROJECT_VERSION'."
    exit 0
fi
echo "Found $num_categories categories."
echo ""

# --- Iterate through each category ---
echo "$categories_json" | jq -c '.[]' | while IFS= read -r category_item; do
    category_type=$(echo "$category_item" | jq -r '.type')
    category_slug=$(echo "$category_item" | jq -r '.slug')
    category_title=$(echo "$category_item" | jq -r '.title')
    
    echo "--------------------------------------------------"
    echo "Processing category: '$category_title' (Type: ${category_type}, Slug: $category_slug)"

    category_docs_dir="$OUTPUT_DIR/$category_type/$category_slug"
    mkdir -p "$category_docs_dir"
    category_image_storage_dir="$category_docs_dir/images"
    mkdir -p "$category_image_storage_dir"

    # --- Get documents in the category ---
    echo "  Fetching documents in category '$category_slug'..."
    docs_in_category_json=$(curl -s -H "x-readme-version: $PROJECT_VERSION" -u "$README_API_KEY:" "$README_API_URL/categories/$category_slug/docs")

    if [ -z "$docs_in_category_json" ]; then
        warning_msg "  Could not fetch documents for category '$category_slug', or the category is empty."
        continue
    fi
    if echo "$docs_in_category_json" | jq -e 'if type == "object" then .error else null end' | grep -q -v null; then
        warning_msg "  API response for document list in category '$category_slug' contains an error: $(echo "$docs_in_category_json" | jq '.') "
        continue
    fi

    num_docs_in_category=$(echo "$docs_in_category_json" | jq '. | length')
    if [ "$num_docs_in_category" -eq 0 ]; then
        echo "  Note: No documents in category '$category_slug'."
        continue
    fi
    echo "  Found $num_docs_in_category documents."

    # --- Iterate through each document in the category ---
    echo "$docs_in_category_json" | jq -c '.[]' | while IFS= read -r doc_item_summary; do
        doc_slug=$(echo "$doc_item_summary" | jq -r '.slug')
        doc_title=$(echo "$doc_item_summary" | jq -r '.title')
        
        echo "    ------------------------------------"
        echo "    Processing document: '$doc_title' (Slug: $doc_slug)"
        
        # Get full document content
        doc_content_json=$(curl -s -H "x-readme-version: $PROJECT_VERSION" -u "$README_API_KEY:" "$README_API_URL/docs/$doc_slug")

        if [ -z "$doc_content_json" ]; then
            warning_msg "      Could not fetch content for document '$doc_slug' (empty response)."
            continue
        fi

        is_hidden=$(echo "$doc_content_json" | jq -r '.hidden') 
        if [ "$is_hidden" == "true" ] && [ "$FINAL_EXPORT_HIDDEN_DOCS" != "true" ]; then
            echo "      Skipping hidden document: '$doc_title' (EXPORT_HIDDEN_DOCS is not set to 'true')"
            continue
        fi

        markdown_body=$(echo "$doc_content_json" | jq -r '.body')
        if echo "$doc_content_json" | jq -e 'if type == "object" then .error else null end' | grep -q -v null; then
            warning_msg "      API response for document '$doc_slug' content contains an .error field, but attempting to process .body. Error details: $(echo "$doc_content_json" | jq -c '.error')"
        fi

        if [ "$markdown_body" == "null" ] || [ -z "$markdown_body" ]; then
            warning_msg "      Document '$doc_slug' has empty .body content."
            markdown_body="" # Ensure markdown_body is not null
        fi
        processed_markdown="$markdown_body"

        # --- Parse and download images ---
        echo "      Parsing and downloading images..."
        temp_img_list_file=$(mktemp) # Create temporary file to store image URLs

        # Use awk to extract image URLs from markdown content
        # Logic: Find `![]()` format and extract the URL within
        echo "$markdown_body" | awk '
        {
            line = $0
            while ( (start_alt = index(line, "![")) > 0 ) {
                end_alt = index(substr(line, start_alt), "]")
                if (end_alt == 0) { break } 
                
                if (substr(line, start_alt + end_alt, 1) == "(") {
                    link_content_start = start_alt + end_alt + 1
                    link_content_end_paren = index(substr(line, link_content_start), ")")
                    if (link_content_end_paren == 0) { break } 
                    
                    full_link_content = substr(line, link_content_start, link_content_end_paren - 1)
                    img_url = ""
                    space_pos = index(full_link_content, " ") # Handle title in ![alt](url "title")
                    
                    if (space_pos > 0) {
                        img_url = substr(full_link_content, 1, space_pos - 1)
                    } else {
                        img_url = full_link_content
                    }
                    
                    if (img_url ~ /^https?:\/\//) { # Only process http/https URLs
                        print img_url
                    }
                    line = substr(line, link_content_start + link_content_end_paren)
                } else {
                    line = substr(line, start_alt + 2) 
                }
            }
        }' | sort -u > "$temp_img_list_file"

        images_downloaded_count=0
        while IFS= read -r img_url; do
            if [[ -z "$img_url" ]]; then
                continue # Skip empty URLs
            fi

            img_filename_raw=$(basename "$img_url")
            img_filename_base="${img_filename_raw%%\?*}" # Remove URL query parameters
            img_filename_sane=$(echo "$img_filename_base" | sed 's/[^a-zA-Z0-9._-]/_/g') # Sanitize filename

            # If sanitized filename is empty (e.g., URL ends with /), use a hash
            if [ -z "$img_filename_sane" ]; then
                img_hash=$(echo -n "$img_url" | md5sum | cut -d' ' -f1)
                img_filename_sane="image_${img_hash}.png" # Default to .png
            fi
            
            # Ensure file has an extension
            if [[ ! "$img_filename_sane" == *.* ]]; then
                original_extension="${img_url##*.}"
                original_extension="${original_extension%%\?*}" # Remove query params from extension
                if [[ "$original_extension" =~ ^(jpg|jpeg|png|gif|svg|webp|ico)$ ]]; then
                    img_filename_sane="${img_filename_sane}.${original_extension}"
                else
                    img_filename_sane="${img_filename_sane}.png" # Default to .png for unknown extensions
                fi
            fi

            local_img_path_relative_to_md="images/$img_filename_sane"
            local_img_download_path="$category_image_storage_dir/$img_filename_sane"

            echo "        Downloading image: $img_url -> $local_img_download_path"
            if curl -s -L "$img_url" -o "$local_img_download_path"; then
                # Use perl to replace image paths in markdown, avoids issues with parentheses in sed
                # \Q$img_url\E treats special characters in $img_url as literals
                # [^)]* matches any characters after the URL up to the closing parenthesis (e.g. image title)
                processed_markdown=$(perl -e '
                    $markdown = do { local $/; <STDIN> };
                    $img_url = $ARGV[0];
                    $new_path = $ARGV[1];
                    $markdown =~ s/!\[([^\]]*)\]\(\Q$img_url\E[^\)]*\)/![$1]($new_path)/g;
                    print $markdown;
                ' <<<"$processed_markdown" "$img_url" "$local_img_path_relative_to_md")
                images_downloaded_count=$((images_downloaded_count + 1))
            else
                warning_msg "        Failed to download image $img_url"
            fi
        done < "$temp_img_list_file"
        rm "$temp_img_list_file" # Delete temporary file
        if [ "$images_downloaded_count" -gt 0 ]; then
            echo "        Successfully downloaded $images_downloaded_count images."
        else
            echo "        No images found or downloaded for this document."
        fi


        # --- Save Markdown file ---
        output_md_file="${category_docs_dir}/${doc_slug}.md"
        echo "      Saving Markdown to: $output_md_file"
        printf "%s" "$processed_markdown" >"$output_md_file"
        if [ $? -ne 0 ]; then
            warning_msg "      Failed to save Markdown file $output_md_file."
        fi
        echo "    Document '$doc_title' (Slug: $doc_slug) processing complete."
        
    done
    echo "  All processable documents in category '$category_title' (Slug: $category_slug) have been processed."
    echo ""
done

echo "=================================================="
echo "All operations completed!"
echo "Documents have been saved in subdirectories under '$OUTPUT_DIR'."
echo "=================================================="

exit 0
