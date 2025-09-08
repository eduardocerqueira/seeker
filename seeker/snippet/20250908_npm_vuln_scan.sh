#date: 2025-09-08T17:02:59Z
#url: https://api.github.com/gists/df3ca42414cf4457d72d3859f3d084e4
#owner: https://api.github.com/users/renatonascalves

#!/bin/bash

# Simplified Malicious NPM Package Security Scanner
# Scans package-lock.json files for known malicious packages

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Malicious packages and versions from security bulletin
MALICIOUS_PACKAGES=(
    "backslash.*0\.2\.1"
    "chalk-template.*1\.1\.1"
    "supports-hyperlinks.*4\.1\.1"
    "has-ansi.*6\.0\.1"
    "simple-swizzle.*0\.2\.3"
    "color-string.*2\.1\.1"
    "error-ex.*1\.3\.3"
    "color-name.*2\.0\.1"
    "is-arrayish.*0\.3\.3"
    "slice-ansi.*7\.1\.1"
    "color-convert.*3\.1\.1"
    "wrap-ansi.*9\.0\.1"
    "ansi-regex.*6\.2\.1"
    "supports-color.*10\.2\.1"
    "strip-ansi.*7\.1\.1"
    "chalk.*5\.6\.1"
    "debug.*4\.4\.2"
    "ansi-styles.*6\.2\.2"
)

# Arrays to store results
declare -a INFECTED_FILES=()
declare -a CLEAN_FILES=()

scan_directory() {
    local scan_dir="${1:-.}"
    
    if [[ ! -d "$scan_dir" ]]; then
        echo -e "${RED}Error: Directory '$scan_dir' does not exist.${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}🔍 Scanning for malicious npm packages in: $(realpath "$scan_dir")${NC}"
    echo "Searching for files to scan (this might take a bit)..."
    
    # Find all package-lock.json files and scan them
    while IFS= read -r -d '' lockfile; do
        local found_in_file=false
        local vulnerabilities=()
        
        for pattern in "${MALICIOUS_PACKAGES[@]}"; do
            if grep -q "\"$pattern\"" "$lockfile"; then
                local matches=$(grep "\"$pattern\"" "$lockfile")
                vulnerabilities+=("$matches")
                found_in_file=true
            fi
        done
        
        if [[ "$found_in_file" == true ]]; then
            echo -n "❌"
            INFECTED_FILES+=("$lockfile")
            # Store vulnerabilities for this file
            for vuln in "${vulnerabilities[@]}"; do
                INFECTED_FILES+=("  $vuln")
            done
        else
            echo -n "."
            CLEAN_FILES+=("$lockfile")
        fi
        
    done < <(find "$scan_dir" -name "package-lock.json" -type f -print0)
    
    echo "" # New line after progress
    echo ""
    
    # Summary
    echo "================================================"
    echo -e "${BLUE}SCAN SUMMARY${NC}"
    echo "================================================"
    echo "Total package-lock.json files scanned: $((${#INFECTED_FILES[@]} / 2 + ${#CLEAN_FILES[@]}))"
    echo "Clean files: ${#CLEAN_FILES[@]}"
    echo "Infected files: $((${#INFECTED_FILES[@]} / 2))"
    echo ""
    
    if [[ ${#INFECTED_FILES[@]} -gt 0 ]]; then
        echo -e "${RED}🚨 SECURITY ALERT: Malicious packages detected!${NC}"
        echo ""
        echo -e "${YELLOW}INFECTED FILES AND VULNERABILITIES:${NC}"
        
        local current_file=""
        for item in "${INFECTED_FILES[@]}"; do
            if [[ "$item" != "  "* ]]; then
                # This is a filename
                current_file="$item"
                echo -e "${RED}❌ $current_file${NC}"
            else
                # This is a vulnerability line
                echo "$item"
            fi
        done
        
        echo ""
    else
        echo -e "${GREEN}✅ All files are clean! No malicious packages detected.${NC}"
    fi
    
    echo "================================================"
    exit $((${#INFECTED_FILES[@]} / 2))
}

# Show usage if needed
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [directory_to_scan]"
    echo "  directory_to_scan: Directory to scan (default: current directory)"
    exit 0
fi

scan_directory "$1"

