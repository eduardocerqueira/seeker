#date: 2025-12-15T17:17:15Z
#url: https://api.github.com/gists/06cf32400b41cd40019a884283baf4de
#owner: https://api.github.com/users/xrendan

#!/bin/bash

# Upload Financial Statements Script
# Usage: ./upload_financial_statements.sh <directory> <api_key> [api_url]
#
# Directory structure: /path/to/dir/<census_subdivision_id>/<filename_with_year>.pdf
# Example: /data/statements/3520005/financial_statement_2023.pdf

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_info() {
    echo -e "${BLUE}INFO: $1${NC}"
}

# Function to extract year from filename
# If multiple years found, returns the latest year
extract_year() {
    local filename="$1"
    local max_year=0

    # Find all 4-digit years (19xx or 20xx) in the filename
    while [[ $filename =~ (19|20)[0-9]{2} ]]; do
        local year="${BASH_REMATCH[0]}"

        # Update max_year if this year is greater
        if [ "$year" -gt "$max_year" ]; then
            max_year="$year"
        fi

        # Remove the matched year from filename to find next occurrence
        filename="${filename/${BASH_REMATCH[0]}/}"
    done

    # Return the maximum year found
    if [ "$max_year" -gt 0 ]; then
        echo "$max_year"
        return 0
    else
        return 1
    fi
}

# Function to upload a single file
upload_file() {
    local census_subdivision_id="$1"
    local year="$2"
    local file_path="$3"
    local api_key="$4"
    local api_url="$5"

    local endpoint="${api_url}/api/v1/bodies/${census_subdivision_id}/${year}"

    print_info "Uploading: CSD=${census_subdivision_id}, Year=${year}, File=$(basename "$file_path")"

    # Make the API call
    response=$(curl -s -w "\n%{http_code}" -X POST \
        "$endpoint" \
        -H "Authorization: Bearer ${api_key}" \
        -F "document=@${file_path}")

    # Extract HTTP status code (last line)
    http_code=$(echo "$response" | tail -n1)
    # Extract response body (all but last line)
    response_body=$(echo "$response" | sed '$d')

    case $http_code in
        201)
            print_success "Uploaded successfully"
            echo "$response_body" | jq -r '.message' 2>/dev/null || echo "$response_body"
            return 0
            ;;
        400)
            print_error "Bad request"
            echo "$response_body" | jq -r '.error + ": " + .details' 2>/dev/null || echo "$response_body"
            return 1
            ;;
        401)
            print_error "Unauthorized - Invalid API key"
            echo "$response_body" | jq -r '.error' 2>/dev/null || echo "$response_body"
            return 1
            ;;
        404)
            print_error "Body not found for CSD ${census_subdivision_id}"
            echo "$response_body" | jq -r '.error' 2>/dev/null || echo "$response_body"
            return 1
            ;;
        409)
            print_warning "Financial statement already exists for ${year}"
            echo "$response_body" | jq -r '.details' 2>/dev/null || echo "$response_body"
            return 0
            ;;
        *)
            print_error "Unexpected error (HTTP ${http_code})"
            echo "$response_body"
            return 1
            ;;
    esac
}

# Main script

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <directory> <api_key> [api_url]"
    echo ""
    echo "Arguments:"
    echo "  directory  - Path to directory containing financial statements"
    echo "               Structure: <directory>/<census_subdivision_id>/<filename_with_year>.pdf"
    echo "  api_key    - Your Build Canada Hub API token"
    echo "  api_url    - (Optional) API base URL (default: https://hub.buildcanada.com)"
    echo ""
    echo "Example:"
    echo "  $0 /data/statements bch_abc123... http://localhost:3000"
    exit 1
fi

DIRECTORY="$1"
API_KEY="$2"
API_URL="${3:-https://hub.buildcanada.com}"

# Validate directory exists
if [ ! -d "$DIRECTORY" ]; then
    print_error "Directory does not exist: $DIRECTORY"
    exit 1
fi

# Validate API key format
if [[ ! $API_KEY =~ ^bch_ ]]; then
    print_warning "API key does not start with 'bch_' - are you sure this is correct?"
fi

# Check for required commands
for cmd in curl jq; do
    if ! command -v $cmd &> /dev/null; then
        print_error "Required command '$cmd' not found. Please install it."
        exit 1
    fi
done

print_info "Starting upload process..."
print_info "Directory: $DIRECTORY"
print_info "API URL: $API_URL"
echo ""

# Counters
total_files=0
successful_uploads=0
failed_uploads=0
skipped_files=0

# Find all PDF files in subdirectories
while IFS= read -r -d '' pdf_file; do
    ((total_files++))

    # Extract census_subdivision_id from directory name
    census_subdivision_id=$(basename "$(dirname "$pdf_file")")

    # Extract filename
    filename=$(basename "$pdf_file")

    # Extract year from filename
    if year=$(extract_year "$filename"); then
        echo ""
        echo "[$total_files] Processing: $pdf_file"

        if upload_file "$census_subdivision_id" "$year" "$pdf_file" "$API_KEY" "$API_URL"; then
            ((successful_uploads++))
        else
            ((failed_uploads++))
        fi
    else
        print_warning "Could not extract year from filename: $filename (skipping)"
        ((skipped_files++))
    fi

done < <(find "$DIRECTORY" -type f -name "*.pdf" -print0)

# Print summary
echo ""
echo "================================================"
echo "                   SUMMARY                      "
echo "================================================"
echo "Total files found:      $total_files"
echo "Successful uploads:     $successful_uploads"
echo "Failed uploads:         $failed_uploads"
echo "Skipped files:          $skipped_files"
echo "================================================"

if [ $failed_uploads -gt 0 ]; then
    exit 1
else
    exit 0
fi
