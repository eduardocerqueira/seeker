#date: 2026-01-12T17:20:11Z
#url: https://api.github.com/gists/72e54c9d18b33677669a52eb048f0bc4
#owner: https://api.github.com/users/gbhat618

#!/bin/bash

# Script to build and verify Java versions in all SSH agent Docker images
# This script builds each agent's Dockerfile and checks the Java version

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

AGENTS_DIR="src/test/resources/io/jenkins/plugins/sshbuildagents/ssh/agents"
BASE_DIR=$(pwd)
TEMP_TAG_PREFIX="ssh-agent-test"
RESULTS_FILE="java-version-results.txt"

# Check if we're in the right directory
if [ ! -d "$AGENTS_DIR" ]; then
    echo -e "${RED}Error: Cannot find agents directory at $AGENTS_DIR${NC}"
    echo "Please run this script from the root of the ssh-agents-plugin repository"
    exit 1
fi

# Clear previous results
> "$RESULTS_FILE"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SSH Agents Java Version Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# First, build the base image if it exists
if [ -d "$AGENTS_DIR/ssh-agent-base" ]; then
    echo -e "${YELLOW}Building base image first...${NC}"
    cd "$AGENTS_DIR/ssh-agent-base"
    docker build -t ghcr.io/jenkinsci/ssh-agents-plugin:base48d6d44 . > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Base image built successfully${NC}"
    else
        echo -e "${RED}✗ Failed to build base image${NC}"
        exit 1
    fi
    cd "$BASE_DIR"
    echo ""
fi

# Counter for summary
total=0
success=0
failed=0
java21=0
java17=0
other=0

# Iterate through all subdirectories
for agent_dir in "$AGENTS_DIR"/*/; do
    if [ -d "$agent_dir" ]; then
        agent_name=$(basename "$agent_dir")

        # Skip if no Dockerfile exists
        if [ ! -f "$agent_dir/Dockerfile" ]; then
            continue
        fi

        total=$((total + 1))

        echo -e "${BLUE}[$total] Processing: $agent_name${NC}"

        # Build the Docker image
        temp_tag="$TEMP_TAG_PREFIX:$agent_name"
        echo -n "  Building image... "

        cd "$agent_dir"
        if docker build -t "$temp_tag" . > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"

            # Run container and get Java version
            echo -n "  Checking Java version... "
            java_version=$(docker run --rm "$temp_tag" java -version 2>&1 | head -n 1)

            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓${NC}"
                success=$((success + 1))

                # Parse Java version
                if echo "$java_version" | grep -q "openjdk version \"21"; then
                    echo -e "  ${GREEN}Java Version: $java_version${NC}"
                    java21=$((java21 + 1))
                    echo "$agent_name: ✓ Java 21 - $java_version" >> "$BASE_DIR/$RESULTS_FILE"
                elif echo "$java_version" | grep -q "openjdk version \"17"; then
                    echo -e "  ${RED}Java Version: $java_version${NC}"
                    java17=$((java17 + 1))
                    echo "$agent_name: ✗ Java 17 - $java_version" >> "$BASE_DIR/$RESULTS_FILE"
                else
                    echo -e "  ${YELLOW}Java Version: $java_version${NC}"
                    other=$((other + 1))
                    echo "$agent_name: ? Other - $java_version" >> "$BASE_DIR/$RESULTS_FILE"
                fi
            else
                echo -e "${RED}✗ (Java not found or container failed)${NC}"
                failed=$((failed + 1))
                echo "$agent_name: ✗ Failed to get Java version" >> "$BASE_DIR/$RESULTS_FILE"
            fi

            # Clean up the temporary image
            docker rmi "$temp_tag" > /dev/null 2>&1
        else
            echo -e "${RED}✗ (Build failed)${NC}"
            failed=$((failed + 1))
            echo "$agent_name: ✗ Build failed" >> "$BASE_DIR/$RESULTS_FILE"
        fi

        cd "$BASE_DIR"
        echo ""
    fi
done

# Print summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total agents checked: $total"
echo -e "${GREEN}Successfully checked: $success${NC}"
echo -e "${RED}Failed: $failed${NC}"
echo ""
echo -e "${GREEN}Java 21: $java21${NC}"
echo -e "${RED}Java 17: $java17${NC}"
if [ $other -gt 0 ]; then
    echo -e "${YELLOW}Other versions: $other${NC}"
fi
echo ""
echo -e "Detailed results saved to: ${YELLOW}$RESULTS_FILE${NC}"
echo ""

# Clean up base image
docker rmi ghcr.io/jenkinsci/ssh-agents-plugin:base48d6d44 > /dev/null 2>&1 || true

if [ $java21 -eq $total ] && [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ All agents are using Java 21!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some agents are not using Java 21 or failed to build.${NC}"
    exit 1
fi
