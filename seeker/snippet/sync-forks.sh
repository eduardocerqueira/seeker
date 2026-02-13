#date: 2026-02-13T17:23:07Z
#url: https://api.github.com/gists/f9c00d59252a26cc889ea69e362b546f
#owner: https://api.github.com/users/djdanielsson

#!/bin/bash

# Script to sync all GitHub forks using GitHub's API (no cloning needed!)
# This uses the same "Sync fork" button you see in the GitHub web UI

# Note: Don't use set -e here so we continue processing all repos even if some fail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) is not installed."
        log_info "Install it: sudo dnf install gh"
        exit 1
    fi

    if ! gh auth status &> /dev/null; then
        log_error "GitHub CLI is not authenticated."
        log_info "Run: gh auth login"
        exit 1
    fi
}

sync_fork() {
    local repo_full_name="$1"
    local default_branch="$2"
    
    log_info "Syncing ${repo_full_name}..."
    
    # Use GitHub API to sync the fork (same as clicking "Sync fork" button)
    # This merges upstream changes into the fork without cloning
    local response
    local exit_code
    
    # Capture the API response (both stdout and stderr)
    response=$(gh api \
        --method POST \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "/repos/${repo_full_name}/merge-upstream" \
        -f branch="${default_branch}" 2>&1 || true)
    
    exit_code=$?
    
    # Handle empty responses
    if [ -z "$response" ]; then
        log_error "✗ Failed to sync ${repo_full_name} - empty response from GitHub API"
        return 1
    fi
    
    # Check for specific error conditions FIRST (before checking exit code)
    # These errors have exit_code != 0 but give us useful info
    
    # 1. Merge conflicts (HTTP 409)
    if echo "$response" | grep -qi "There are merge conflicts\|status.*409"; then
        log_warning "⚠ ${repo_full_name} has merge conflicts - needs manual resolution"
        return 3  # conflict
    fi
    
    # 2. OAuth workflow scope issue (HTTP 422)
    if echo "$response" | grep -qi "refusing to allow an OAuth App.*workflow.*scope\|status.*422"; then
        log_error "✗ ${repo_full_name} - OAuth workflow scope missing"
        log_info "  Fix: Run 'gh auth refresh -s workflow' to grant workflow scope"
        return 1  # failed
    fi
    
    # 3. Not found or deleted upstream (HTTP 404)
    if echo "$response" | grep -qi "not found\|status.*404"; then
        log_warning "⚠ ${repo_full_name} - upstream not found or repo deleted"
        return 4  # error
    fi
    
    # 4. Access/permission errors
    if echo "$response" | grep -qi "Could not resolve to a Repository\|Forbidden\|status.*403"; then
        log_error "✗ ${repo_full_name} - repository access/permission error"
        return 1  # failed
    fi
    
    # Now check if the operation succeeded
    if [ $exit_code -eq 0 ]; then
        # Check if there was actually something to merge
        if echo "$response" | grep -q '"merge_type"'; then
            # Check what type of merge
            if echo "$response" | grep -q '"merge_type".*"none"'; then
                log_success "✓ ${repo_full_name} is already up to date"
                return 2  # up to date
            else
                log_success "✓ Synced ${repo_full_name}"
                return 0  # synced
            fi
        else
            log_success "✓ ${repo_full_name} is already up to date"
            return 2  # up to date
        fi
    else
        # Exit code != 0, but we didn't catch the error above
        # This is a generic failure
        log_error "✗ Failed to sync ${repo_full_name}"
        # Only show first 200 chars of error to avoid spam
        local error_msg=$(echo "$response" | head -c 200)
        log_error "Error: ${error_msg}..."
        return 1  # failed
    fi
}

main() {
    log_info "GitHub Fork Sync (API-based - no cloning needed!)"
    echo ""
    
    # Check requirements
    check_requirements
    
    # Get list of forked repositories
    log_info "Fetching your forked repositories..."
    repos_json=$(gh repo list --fork --json nameWithOwner,defaultBranchRef --limit 1000)
    
    if [ -z "$repos_json" ] || [ "$repos_json" = "[]" ]; then
        log_warning "No forked repositories found."
        exit 0
    fi
    
    # Count repositories
    repo_count=$(echo "$repos_json" | jq '. | length')
    log_info "Found ${repo_count} forked repositories"
    echo ""
    
    # Track statistics and lists
    success_count=0
    fail_count=0
    uptodate_count=0
    conflict_count=0
    
    declare -a synced_repos=()
    declare -a uptodate_repos=()
    declare -a conflict_repos=()
    declare -a failed_repos=()
    
    # Process each repository
    while IFS= read -r repo; do
        # Wrap in error handling to continue even if parsing fails
        repo_name=$(echo "$repo" | jq -r '.nameWithOwner' 2>/dev/null || echo "unknown")
        default_branch=$(echo "$repo" | jq -r '.defaultBranchRef.name // "main"' 2>/dev/null || echo "main")
        
        # Skip if we couldn't parse the repo name
        if [ "$repo_name" = "unknown" ] || [ -z "$repo_name" ]; then
            log_warning "⚠ Skipping repository with unparseable name"
            ((fail_count++)) || true
            continue
        fi
        
        # Try to sync the fork and capture the result
        # Note: We don't use || true here because we need the actual exit code
        set +e  # Temporarily disable exit on error for this command
        sync_fork "$repo_name" "$default_branch"
        result=$?
        set -e  # Re-enable (though we don't have set -e globally anymore)
        
        case $result in
            0)  # Successfully synced
                ((success_count++)) || true
                synced_repos+=("$repo_name")
                ;;
            2)  # Already up to date
                ((uptodate_count++)) || true
                uptodate_repos+=("$repo_name")
                ;;
            3)  # Merge conflict
                ((conflict_count++)) || true
                conflict_repos+=("$repo_name")
                ;;
            *)  # Failed
                ((fail_count++)) || true
                failed_repos+=("$repo_name")
                ;;
        esac
    done < <(echo "$repos_json" | jq -c '.[]' 2>/dev/null || echo "[]")
    
    # Summary
    echo ""
    log_info "=========================================="
    log_info "Sync Complete!"
    log_info "=========================================="
    echo ""
    
    # Synced repositories
    if [ $success_count -gt 0 ]; then
        log_success "Successfully synced (${success_count}):"
        for repo in "${synced_repos[@]}"; do
            echo -e "  ${GREEN}✓${NC} https://github.com/${repo}"
        done
        echo ""
    fi
    
    # Up to date repositories
    if [ $uptodate_count -gt 0 ]; then
        log_info "Already up to date (${uptodate_count}):"
        for repo in "${uptodate_repos[@]}"; do
            echo -e "  ${BLUE}•${NC} https://github.com/${repo}"
        done
        echo ""
    fi
    
    # Repositories with conflicts
    if [ $conflict_count -gt 0 ]; then
        log_warning "Merge conflicts - manual resolution required (${conflict_count}):"
        for repo in "${conflict_repos[@]}"; do
            echo -e "  ${YELLOW}⚠${NC} https://github.com/${repo}"
        done
        echo ""
        log_info "To resolve conflicts manually:"
        echo -e "  1. Go to the repository on GitHub (click links above)"
        echo -e "  2. Click 'Sync fork' button"
        echo -e "  3. Choose 'Discard commits' to replace with upstream, or resolve locally"
        echo ""
    fi
    
    # Failed repositories
    if [ $fail_count -gt 0 ]; then
        log_error "Failed to sync (${fail_count}):"
        for repo in "${failed_repos[@]}"; do
            echo -e "  ${RED}✗${NC} https://github.com/${repo}"
        done
        echo ""
        
        # Check if OAuth scope issue is likely
        if grep -q "OAuth.*workflow.*scope" /tmp/sync-fork-errors.log 2>/dev/null; then
            log_info "Some failures may be due to missing GitHub OAuth workflow scope."
            log_info "To fix, run: ${YELLOW}gh auth refresh -s workflow${NC}"
            echo -e "  This will grant permission to update GitHub Actions workflows in forks."
            echo ""
        fi
    fi
    
    # Final stats
    log_info "Summary: ${success_count} synced, ${uptodate_count} up-to-date, ${conflict_count} conflicts, ${fail_count} failed"
    echo ""
    log_info "Note: All changes are on GitHub. Run 'git pull' in local repos to get updates."
}

# Run main function
main
