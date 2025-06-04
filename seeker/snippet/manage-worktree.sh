#date: 2025-06-04T16:56:25Z
#url: https://api.github.com/gists/b1717507a3cc914086e2a40316736d80
#owner: https://api.github.com/users/jamesaphoenix

#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function show_help {
    echo -e "${BLUE}OctoSpark Worktree Manager${NC}"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  create <branch-name>     Create a new worktree with schema isolation"
    echo "  list                     List all worktrees and their schemas"
    echo "  switch <name>            Switch to a worktree directory"
    echo "  remove <name>            Remove a worktree and its schema"
    echo "  help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 create feature-auth"
    echo "  $0 list"
    echo "  $0 switch feature-auth"
    echo "  $0 remove feature-auth"
}

function create_worktree {
    local BRANCH_NAME=$1
    if [ -z "$BRANCH_NAME" ]; then
        echo -e "${RED}❌ Please provide a branch name${NC}"
        echo "Usage: $0 create <branch-name>"
        exit 1
    fi

    local WORKTREE_PATH="worktrees/$BRANCH_NAME"
    
    # Check if worktree already exists
    if [ -d "$WORKTREE_PATH" ]; then
        echo -e "${RED}❌ Worktree already exists at: $WORKTREE_PATH${NC}"
        exit 1
    fi

    echo -e "${GREEN}🌳 Creating worktree: $BRANCH_NAME${NC}"
    
    # Create the worktree
    git worktree add "$WORKTREE_PATH" -b "$BRANCH_NAME" 2>/dev/null || {
        # If branch already exists, just check it out
        git worktree add "$WORKTREE_PATH" "$BRANCH_NAME"
    }

    echo -e "${GREEN}✅ Worktree created at: $WORKTREE_PATH${NC}"
    echo ""
    echo -e "${YELLOW}Next steps (run from this directory):${NC}"
    echo -e "1. ./scripts/setup-worktree-root.sh $WORKTREE_PATH"
    echo -e "2. $WORKTREE_PATH/run-migrations.sh"
    echo -e "3. $WORKTREE_PATH/seed-worktree.sh"
    echo -e "4. cd $WORKTREE_PATH && pnpm dev (or use $WORKTREE_PATH/dev-worktree.sh)"
}

function list_worktrees {
    echo -e "${BLUE}📋 Active Worktrees:${NC}"
    echo ""
    
    # List git worktrees
    git worktree list | while read -r line; do
        local path=$(echo "$line" | awk '{print $1}')
        local branch=$(echo "$line" | awk '{print $3}' | tr -d '[]')
        local worktree_name=$(basename "$path")
        
        # Skip the main worktree
        if [ "$worktree_name" == "octospark-services" ]; then
            continue
        fi
        
        # Generate schema name
        local schema="wt_$(echo $worktree_name | tr '-' '_' | tr '[:upper:]' '[:lower:]')"
        
        echo -e "${GREEN}$worktree_name${NC}"
        echo -e "  Path: $path"
        echo -e "  Branch: $branch"
        echo -e "  Schema: $schema"
        
        # Check if schema exists in database
        if psql postgresql://postgres:postgres@localhost:54322/postgres -tAc "SELECT 1 FROM information_schema.schemata WHERE schema_name = '$schema'" 2>/dev/null | grep -q 1; then
            echo -e "  Database: ${GREEN}✓ Schema exists${NC}"
        else
            echo -e "  Database: ${YELLOW}⚠ Schema not created${NC}"
        fi
        
        # Check Docker containers
        local web_container="octospark-web-$worktree_name"
        local api_container="octospark-api-$worktree_name"
        
        if docker ps --format "{{.Names}}" | grep -q "^$web_container$"; then
            local web_port=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep "^$web_container" | grep -oE '0.0.0.0:[0-9]+' | head -1 | cut -d: -f2)
            echo -e "  Web: ${GREEN}✓ Running${NC} on port ${web_port:-unknown}"
        else
            echo -e "  Web: ${YELLOW}○ Not running${NC}"
        fi
        
        if docker ps --format "{{.Names}}" | grep -q "^$api_container$"; then
            local api_port=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep "^$api_container" | grep -oE '0.0.0.0:[0-9]+' | head -1 | cut -d: -f2)
            echo -e "  API: ${GREEN}✓ Running${NC} on port ${api_port:-unknown}"
        else
            echo -e "  API: ${YELLOW}○ Not running${NC}"
        fi
        
        echo ""
    done
}

function switch_worktree {
    local WORKTREE_NAME=$1
    if [ -z "$WORKTREE_NAME" ]; then
        echo -e "${RED}❌ Please provide a worktree name${NC}"
        echo "Usage: $0 switch <name>"
        exit 1
    fi

    local WORKTREE_PATH="worktrees/$WORKTREE_NAME"
    
    if [ ! -d "$WORKTREE_PATH" ]; then
        echo -e "${RED}❌ Worktree not found: $WORKTREE_NAME${NC}"
        echo "Available worktrees:"
        list_worktrees
        exit 1
    fi

    echo -e "${GREEN}Switching to worktree: $WORKTREE_NAME${NC}"
    echo "cd $WORKTREE_PATH"
    echo ""
    echo -e "${YELLOW}Run this command to switch:${NC}"
    echo -e "${BLUE}cd $WORKTREE_PATH${NC}"
}

function remove_worktree {
    local WORKTREE_NAME=$1
    if [ -z "$WORKTREE_NAME" ]; then
        echo -e "${RED}❌ Please provide a worktree name${NC}"
        echo "Usage: $0 remove <name>"
        exit 1
    fi

    local WORKTREE_PATH="worktrees/$WORKTREE_NAME"
    local SCHEMA="wt_$(echo $WORKTREE_NAME | tr '-' '_' | tr '[:upper:]' '[:lower:]')"
    
    if [ ! -d "$WORKTREE_PATH" ]; then
        echo -e "${RED}❌ Worktree not found: $WORKTREE_NAME${NC}"
        exit 1
    fi

    # Get the branch name for this worktree
    local BRANCH_NAME=""
    local worktree_info=$(git worktree list | grep "$WORKTREE_PATH")
    if [ -n "$worktree_info" ]; then
        BRANCH_NAME=$(echo "$worktree_info" | awk '{print $3}' | tr -d '[]')
    fi

    echo -e "${YELLOW}⚠️  This will remove:${NC}"
    echo -e "  - Worktree at: $WORKTREE_PATH"
    echo -e "  - Database schema: $SCHEMA"
    if [ -n "$BRANCH_NAME" ]; then
        echo -e "  - Git branch: $BRANCH_NAME"
    fi
    
    # Check for running Docker containers
    local web_container="octospark-web-$WORKTREE_NAME"
    local api_container="octospark-api-$WORKTREE_NAME"
    local running_containers=false
    
    if docker ps --format "{{.Names}}" | grep -q "^$web_container$"; then
        echo -e "  - Docker container: $web_container"
        running_containers=true
    fi
    
    if docker ps --format "{{.Names}}" | grep -q "^$api_container$"; then
        echo -e "  - Docker container: $api_container"
        running_containers=true
    fi
    
    echo ""
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Cancelled${NC}"
        exit 0
    fi

    # Stop Docker containers first
    if [ "$running_containers" = true ]; then
        echo -e "${GREEN}Stopping Docker containers...${NC}"
        docker stop "$web_container" "$api_container" 2>/dev/null || true
        docker rm "$web_container" "$api_container" 2>/dev/null || true
        # Also stop any related containers (like cloud tasks emulator)
        docker ps --format "{{.Names}}" | grep "$WORKTREE_NAME" | xargs -r docker stop 2>/dev/null || true
        docker ps -a --format "{{.Names}}" | grep "$WORKTREE_NAME" | xargs -r docker rm 2>/dev/null || true
    fi

    # Run cleanup script if it exists
    if [ -f "$WORKTREE_PATH/cleanup-worktree.sh" ]; then
        echo -e "${GREEN}Running cleanup script...${NC}"
        (cd "$WORKTREE_PATH" && ./cleanup-worktree.sh)
    else
        # Manual cleanup
        echo -e "${GREEN}Dropping schema: $SCHEMA${NC}"
        psql postgresql://postgres:postgres@localhost:54322/postgres -c "DROP SCHEMA IF EXISTS $SCHEMA CASCADE;" 2>/dev/null || true
    fi

    # Remove git worktree
    echo -e "${GREEN}Removing git worktree...${NC}"
    git worktree remove "$WORKTREE_PATH" --force

    # Ensure the directory is completely removed
    if [ -d "$WORKTREE_PATH" ]; then
        echo -e "${GREEN}Cleaning up worktree directory...${NC}"
        rm -rf "$WORKTREE_PATH"
    fi

    # Delete the associated branch if it exists
    if [ -n "$BRANCH_NAME" ] && git branch | grep -q "^\s*$BRANCH_NAME$"; then
        echo -e "${GREEN}Deleting git branch: $BRANCH_NAME${NC}"
        git branch -D "$BRANCH_NAME" 2>/dev/null || {
            echo -e "${YELLOW}⚠️  Could not delete branch $BRANCH_NAME (it may be merged or already deleted)${NC}"
        }
    elif [ -n "$BRANCH_NAME" ]; then
        echo -e "${YELLOW}ℹ️  Branch $BRANCH_NAME was already deleted or doesn't exist locally${NC}"
    fi

    echo -e "${GREEN}✅ Worktree and associated branch removed successfully${NC}"
}

# Main command handling
case "${1:-help}" in
    create)
        create_worktree "$2"
        ;;
    list)
        list_worktrees
        ;;
    switch)
        switch_worktree "$2"
        ;;
    remove)
        remove_worktree "$2"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac