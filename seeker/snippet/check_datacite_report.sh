#date: 2025-11-21T16:50:48Z
#url: https://api.github.com/gists/765f5532e40c97a0c3beaa3615b7932f
#owner: https://api.github.com/users/DonRichards

#!/bin/bash

set -euo pipefail

# Configuration
DATACITE_KEY="xxxx.xxx"
REPORTS_API="https://api.datacite.org/reports?created_by=${DATACITE_KEY}"
REPORT_DETAIL_API="https://api.datacite.org/reports"
MAX_RETRIES=3
RETRY_DELAY=2
CURL_TIMEOUT=30
CURL_MAX_SIZE=$((100 * 1024 * 1024))  # 100MB limit

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to log messages
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to check dependencies
check_dependencies() {
    local missing_deps=()

    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi

    if ! command -v jq &> /dev/null; then
        missing_deps+=("jq")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_error "Please install them and try again."
        exit 1
    fi
}

# Function to perform curl with retry logic
curl_with_retry() {
    local url="$1"
    local output_file="$2"
    local attempt=1

    while [ $attempt -le $MAX_RETRIES ]; do
        log_info "Fetching $url (attempt $attempt/$MAX_RETRIES)..."

        local http_code
        http_code=$(curl -s -w "%{http_code}" -o "$output_file" \
            --max-time "$CURL_TIMEOUT" \
            --max-filesize "$CURL_MAX_SIZE" \
            --compressed \
            "$url" 2>/dev/null || echo "000")

        if [ "$http_code" -eq 200 ]; then
            # Verify file is not empty
            if [ ! -s "$output_file" ]; then
                log_warn "Received empty response (attempt $attempt/$MAX_RETRIES)"
            else
                log_info "Successfully fetched data (HTTP $http_code)"
                return 0
            fi
        elif [ "$http_code" -eq 000 ]; then
            log_warn "Network error or timeout (attempt $attempt/$MAX_RETRIES)"
        elif [ "$http_code" -eq 429 ]; then
            log_warn "Rate limited (HTTP 429) (attempt $attempt/$MAX_RETRIES)"
            # Longer delay for rate limiting
            if [ $attempt -lt $MAX_RETRIES ]; then
                local delay=$((RETRY_DELAY * attempt * 3))
                log_info "Retrying in ${delay}s..."
                sleep $delay
            fi
            attempt=$((attempt + 1))
            continue
        elif [ "$http_code" -ge 500 ]; then
            log_warn "Server error (HTTP $http_code) (attempt $attempt/$MAX_RETRIES)"
        else
            log_warn "HTTP error $http_code (attempt $attempt/$MAX_RETRIES)"
        fi

        if [ $attempt -lt $MAX_RETRIES ]; then
            local delay=$((RETRY_DELAY * attempt))
            log_info "Retrying in ${delay}s..."
            sleep $delay
        fi

        attempt=$((attempt + 1))
    done

    log_error "Failed to fetch $url after $MAX_RETRIES attempts"
    return 1
}

# Function to calculate last month's first day
calculate_target_date() {
    # Get current year and month (support TEST_DATE env var for testing)
    local current_year
    local current_month

    if [ -n "${TEST_DATE:-}" ]; then
        # Try macOS date format first, then GNU date format
        current_year=$(date -j -f "%Y-%m-%d" "$TEST_DATE" +%Y 2>/dev/null || date -d "$TEST_DATE" +%Y 2>/dev/null || echo "")
        current_month=$(date -j -f "%Y-%m-%d" "$TEST_DATE" +%m 2>/dev/null || date -d "$TEST_DATE" +%m 2>/dev/null || echo "")

        # Validate date parsing succeeded
        if [ -z "$current_year" ] || [ -z "$current_month" ]; then
            log_error "Invalid TEST_DATE format: $TEST_DATE (expected: YYYY-MM-DD)"
            return 1
        fi
    else
        current_year=$(date +%Y)
        current_month=$(date +%m)
    fi

    # Validate year and month are numeric
    if ! [[ "$current_year" =~ ^[0-9]{4}$ ]] || ! [[ "$current_month" =~ ^[0-9]{2}$ ]]; then
        log_error "Invalid date values: year=$current_year, month=$current_month"
        return 1
    fi

    # Calculate last month (handle leading zeros with 10# prefix)
    local target_month=$((10#$current_month - 1))
    local target_year=$current_year

    if [ $target_month -eq 0 ]; then
        target_month=12
        target_year=$((target_year - 1))
    fi

    # Format as YYYY-MM-01
    printf "%04d-%02d-01" $target_year $target_month
}

# Function to validate JSON file
validate_json() {
    local json_file="$1"

    if [ ! -f "$json_file" ]; then
        log_error "JSON file not found: $json_file"
        return 1
    fi

    if ! jq empty "$json_file" 2>/dev/null; then
        log_error "Invalid JSON in file: $json_file"
        return 1
    fi

    return 0
}

# Main script
main() {
    log_info "Starting DataCite report validation"

    # Check dependencies
    check_dependencies

    # Calculate target date (last month's first day)
    if [ -n "${TEST_DATE:-}" ]; then
        log_info "Using TEST_DATE: $TEST_DATE"
    fi
    local target_date
    target_date=$(calculate_target_date)
    log_info "Target begin-date: $target_date"

    # Create temporary files
    local reports_file=$(mktemp) || {
        log_error "Failed to create temporary file for reports"
        exit 1
    }
    local report_detail_file=$(mktemp) || {
        log_error "Failed to create temporary file for report details"
        rm -f "$reports_file"
        exit 1
    }

    # Ensure cleanup on exit (handle multiple signals)
    trap "rm -f '$reports_file' '$report_detail_file'" EXIT INT TERM

    # Step 1: Fetch reports list
    log_info "Step 1: Fetching reports list..."
    if ! curl_with_retry "$REPORTS_API" "$reports_file"; then
        log_error "Failed to fetch reports list"
        exit 1
    fi

    # Validate JSON
    if ! validate_json "$reports_file"; then
        exit 1
    fi

    # Step 2: Find report with matching begin-date
    log_info "Step 2: Searching for report with begin-date=$target_date..."

    # Check if reports array exists and is not empty
    local reports_count
    reports_count=$(jq '.reports | length' "$reports_file" 2>/dev/null || echo "0")
    if [ "$reports_count" -eq 0 ]; then
        log_error "No reports found in API response"
        exit 1
    fi
    log_info "Found $reports_count total report(s) in API"

    local report_id
    report_id=$(jq -r --arg date "$target_date" \
        '.reports[] | select(.["report-header"]["reporting-period"]["begin-date"] == $date) | .id' \
        "$reports_file" 2>/dev/null | head -n 1)

    if [ -z "$report_id" ] || [ "$report_id" == "null" ]; then
        log_error "No report found with begin-date=$target_date"
        log_info "Available begin-dates:"
        jq -r '.reports[].["report-header"]["reporting-period"]["begin-date"]' "$reports_file" 2>/dev/null | sort -u || echo "  (Unable to list dates)"
        exit 1
    fi

    log_info "Found report ID: $report_id"

    # Step 3: Fetch report details
    log_info "Step 3: Fetching report details..."
    local report_url="${REPORT_DETAIL_API}/${report_id}"

    if ! curl_with_retry "$report_url" "$report_detail_file"; then
        log_error "Failed to fetch report details for ID: $report_id"
        exit 1
    fi

    # Validate JSON
    if ! validate_json "$report_detail_file"; then
        exit 1
    fi

    # Step 4: Check for exceptions (warning only)
    log_info "Step 4: Checking for exceptions..."

    local exceptions_count
    exceptions_count=$(jq '[.report["report-header"].exceptions[]? | select(. != {})] | length' "$report_detail_file" 2>/dev/null || echo "0")

    # Validate exceptions_count is a number
    if ! [[ "$exceptions_count" =~ ^[0-9]+$ ]]; then
        log_warn "Unable to parse exceptions count, assuming 0"
        exceptions_count=0
    fi

    if [ "$exceptions_count" -gt 0 ] 2>/dev/null; then
        log_warn "Report contains $exceptions_count exception(s):"
        jq -r '.report["report-header"].exceptions[]? | select(. != {}) | "  Code \(.code // "N/A"): \(.message // "N/A") [Severity: \(.severity // "N/A")]"' "$report_detail_file" 2>/dev/null || log_warn "  (Unable to display exception details)"
    else
        log_info "No exceptions found in report"
    fi

    # Step 5: Validate report-datasets is not empty
    log_info "Step 5: Validating report-datasets..."

    local datasets_count
    datasets_count=$(jq '.report["report-datasets"] | if . == null then 0 else length end' "$report_detail_file" 2>/dev/null || echo "error")

    # Validate datasets_count is a valid number
    if ! [[ "$datasets_count" =~ ^[0-9]+$ ]]; then
        log_error "Report validation FAILED: unable to parse datasets count"
        log_error "Report ID: $report_id"
        log_error "URL: $report_url"
        log_error "Datasets count value: ${datasets_count}"
        exit 1
    fi

    # Check if datasets_count is 0
    if [ "$datasets_count" -eq 0 ]; then
        log_error "Report validation FAILED: report contains ZERO datasets"
        log_error "Report ID: $report_id"
        log_error "URL: $report_url"
        log_error "Datasets count: 0"
        exit 1
    fi

    log_info "Report contains $datasets_count dataset(s)"

    # Success
    echo ""
    log_info "======================================"
    log_info "Report validation SUCCESSFUL"
    log_info "======================================"
    log_info "Report ID: $report_id"
    log_info "Begin Date: $target_date"
    log_info "Datasets: $datasets_count"
    log_info "Exceptions: $exceptions_count"
    log_info "======================================"

    exit 0
}

# Run main function
main
