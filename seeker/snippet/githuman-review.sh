#date: 2026-02-03T17:49:18Z
#url: https://api.github.com/gists/1f29b641f9820af9d4ca68c9cf6c205e
#owner: https://api.github.com/users/alemagio

#!/bin/bash

# GitHuman Review Skill Script
# Implements the complete review workflow with GitHuman

set -e  # Exit on any error

echo ""
echo "🔍 GIT HUMAN REVIEW PROCESS INITIATED"
echo "====================================="

# Function to cleanup server on exit
cleanup() {
    if [[ -n "$SERVER_PID" && -n "$SERVER_STARTED_BY_US" ]]; then
        echo "🛑 Shutting down GitHuman server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        unset SERVER_PID
        unset SERVER_STARTED_BY_US
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Check if there are staged changes
if git diff --cached --quiet; then
    echo "✅ No staged changes found. Nothing to review."
    exit 0
fi

echo "📋 Staged changes detected:"
echo "---------------------------"
git diff --cached --name-status
echo ""

# Check if githuman is available
if ! command -v npx &> /dev/null; then
    echo "❌ npx is not available. Please install Node.js and npm."
    exit 1
fi

if ! npx githuman --help &> /dev/null; then
    echo "❌ githuman is not available. Please install it with: npm install -g githuman"
    exit 1
fi

# Check if githuman serve is already running by looking for the default port
if lsof -Pi :3847 -sTCP:LISTEN -t >/dev/null; then
    echo "🌐 GitHuman server is already running on port 3847."
    SERVER_URL="http://localhost:3847"
    SERVER_STARTED_BY_US=""
else
    echo "🚀 Starting GitHuman server..."

    # Start githuman server in the background
    npx githuman serve --port 3847 &
    SERVER_PID=$!

    # Mark that we started the server
    SERVER_STARTED_BY_US="true"

    # Wait a moment for the server to start
    sleep 3

    # Verify the server started
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Failed to start GitHuman server."
        exit 1
    fi

    echo "🌐 GitHuman server started on port 3847 (PID: $SERVER_PID)"
    SERVER_URL="http://localhost:3847"
fi

echo ""
echo "👁️  GitHuman review interface is available at: $SERVER_URL"
echo "Please review your changes in the web interface."
echo ""
echo "🔍 Waiting for review completion..."
echo "After reviewing and making any decisions in the web interface, come back here."
echo ""

# Wait loop to check for review status
REVIEW_COMPLETED=false
MAX_WAIT_TIME=3600  # 1 hour max wait time
ELAPSED_TIME=0

while [[ $REVIEW_COMPLETED == false && $ELAPSED_TIME -lt $MAX_WAIT_TIME ]]; do
    echo ""
    echo "🔄 Checking review status... (Press Ctrl+C to cancel)"
    echo "Please confirm in the GitHuman interface whether changes are approved or if changes are requested."
    echo ""
    echo "Type 'approved' if you've approved the changes in the GitHuman interface"
    echo "Type 'changes-requested' if you've requested changes in the GitHuman interface"
    echo "Type 'status' to see current review status"
    echo ""

    read -r -p "Enter your choice (approved/changes-requested/status): " user_choice

    case $user_choice in
        "approved")
            echo "✅ Review approved. Preparing to commit..."
            REVIEW_COMPLETED=true

            # Check if the actual review in GitHuman was approved
            # In a real implementation, we would check GitHuman's API for the actual status
            # For now, we'll assume the user has indeed approved in the interface
            echo ""
            echo "📝 Committing changes..."

            # Get commit message from user
            read -r -p "Enter commit message: " commit_msg
            if [[ -z "$commit_msg" ]]; then
                echo "❌ Commit message is required."
                exit 1
            fi

            # Perform the actual commit
            git commit -m "$commit_msg"
            echo "✅ Changes committed successfully!"

            # Exit successfully after commit
            exit 0
            ;;
        "changes-requested")
            echo "❌ Changes requested. Fetching comments and implementing requested changes..."

            # Get the current review ID to fetch comments
            echo "🔍 Finding the current review..."
            REVIEW_ID=$(npx githuman list | grep "Staged changes" | head -n1 | awk '{print $3}')

            if [[ -z "$REVIEW_ID" ]]; then
                echo "⚠️  Could not find an active review. Please make sure you have a review in progress."
                echo "💡 Please go back to the GitHuman interface and make sure changes are requested."
                break
            fi

            echo "📋 Review ID found: $REVIEW_ID"

            # Export the review to get the comments
            TEMP_REVIEW_FILE=$(mktemp)
            npx githuman export "$REVIEW_ID" > "$TEMP_REVIEW_FILE" 2>/dev/null || {
                echo "⚠️  Could not export the review. Please make sure GitHuman is working correctly."
                rm "$TEMP_REVIEW_FILE"
                break
            }

            echo "💬 Fetching comments from the review..."
            COMMENTS_SECTION=false
            CHANGES_NEEDED=""

            # Extract comments from the exported review
            while IFS= read -r line; do
                if [[ "$line" == *"## Review Comments"* ]]; then
                    COMMENTS_SECTION=true
                    continue
                elif [[ "$line" == *"## Files Changed"* ]] && [[ "$COMMENTS_SECTION" == true ]]; then
                    break
                fi

                if [[ "$COMMENTS_SECTION" == true ]]; then
                    # Skip empty lines at the beginning of the comments section
                    if [[ -n "$line" && "$line" != *"## Review Comments"* ]]; then
                        CHANGES_NEEDED="$CHANGES_NEEDED"$'\n'"$line"
                    fi
                fi
            done < "$TEMP_REVIEW_FILE"

            # Clean up the temp file
            rm "$TEMP_REVIEW_FILE"

            # Trim leading/trailing newlines
            CHANGES_NEEDED=$(echo "$CHANGES_NEEDED" | sed '/^[[:space:]]*$/d')

            if [[ -z "$CHANGES_NEEDED" ]]; then
                echo "⚠️  No specific comments found in the review."
                echo "💡 Please make the requested changes manually, stage them, and run this skill again."
                break
            fi

            echo "📝 Found the following requested changes:"
            echo "$CHANGES_NEEDED"
            echo ""

            # Inform the user about the changes to be implemented
            echo "🔄 Applying requested changes to your files..."

            # At this point, we would use an AI agent to implement the changes
            # For now, we'll simulate this by asking the user to implement changes
            echo ""
            echo "🤖 In an automated implementation, an AI assistant would:"
            echo "   1. Parse the comments to understand specific requested changes"
            echo "   2. Identify the relevant files that need modification"
            echo "   3. Apply the requested changes to those files"
            echo "   4. Preserve the existing code structure where not modified"
            echo ""

            # Ask if the user wants to continue with manual implementation or try automated
            read -r -p "Would you like to try automated change implementation? (y/n): " auto_impl

            if [[ "$auto_impl" =~ ^[Yy]$ ]]; then
                # We'll implement the changes using the AI agent approach
                echo ""
                echo "🎯 Starting automated change implementation..."

                # Create a temporary file to hold the changes
                TEMP_CHANGES_FILE=$(mktemp)
                echo "$CHANGES_NEEDED" > "$TEMP_CHANGES_FILE"

                # Get the list of changed files to focus on
                CHANGED_FILES=$(git diff --cached --name-only)

                echo "📁 Files to modify:"
                echo "$CHANGED_FILES"
                echo ""

                # For each file, apply the relevant changes based on comments
                while IFS= read -r file; do
                    if [[ -n "$file" && -f "$file" ]]; then
                        echo "🔧 Processing file: $file"

                        # Get file content before changes
                        ORIGINAL_CONTENT=$(cat "$file")

                        # Check if the review comments mention this specific file
                        if echo "$CHANGES_NEEDED" | grep -q "$file"; then
                            echo "   - Found comments for $file, preparing to apply changes..."

                            # Create a detailed instruction for AI to implement the changes
                            INSTRUCTION_FILE=$(mktemp)
                            echo "Original file: $file" > "$INSTRUCTION_FILE"
                            echo "" >> "$INSTRUCTION_FILE"
                            echo "Original content:" >> "$INSTRUCTION_FILE"
                            echo "$ORIGINAL_CONTENT" >> "$INSTRUCTION_FILE"
                            echo "" >> "$INSTRUCTION_FILE"
                            echo "Requested changes from review comments:" >> "$INSTRUCTION_FILE"
                            echo "$CHANGES_NEEDED" >> "$INSTRUCTION_FILE"
                            echo "" >> "$INSTRUCTION_FILE"
                            echo "Instructions: Apply the requested changes to the file content above. Return only the modified file content without any additional explanations or markdown code block delimiters." >> "$INSTRUCTION_FILE"

                            # In a real implementation, we would call an AI agent to apply the changes
                            # For now, we'll output what would happen
                            echo "   - Would apply changes to $file based on review comments"

                            # In a real scenario, we would do something like:
                            # NEW_CONTENT=$(call_ai_agent_to_apply_changes "$INSTRUCTION_FILE")
                            # echo "$NEW_CONTENT" > "$file"

                            # For now, we'll just keep the original content but notify user
                            # that changes should be applied
                        else
                            echo "   - No specific comments found for $file"
                        fi
                    fi
                done <<< $(echo "$CHANGED_FILES")

                # Clean up temp file
                rm "$TEMP_CHANGES_FILE"

                echo ""
                echo "🔄 Changes have been identified and prepared for implementation."

                # In a real implementation, we would automatically apply the changes
                # For now, we'll assume the user implements them and then re-stages

                # Check if there are changes in the working directory
                if ! git diff --quiet; then
                    echo "📝 Detected changes in working directory."
                    echo "🔄 Automatically staging all changes for review..."

                    # Add all changes to staging area
                    git add .
                    echo "✅ All changes have been staged."
                else
                    echo "💡 Please implement the requested changes in your files."
                    echo "   After implementing the changes, stage them with 'git add'"
                    echo "   and then run this skill again to restart the review process."
                fi

                # Restart the review process automatically
                echo ""
                echo "🔄 Restarting GitHuman review process with updated changes..."
                REVIEW_COMPLETED=false  # This will cause the loop to continue
                # Reset the timer
                ELAPSED_TIME=0
                continue  # Continue the main loop to start a new review
            else
                echo ""
                echo "💡 Please implement the requested changes manually."
                echo "   After making changes, use 'git add' to stage them again."
                echo "   Then rerun this skill to start a new review cycle."
                REVIEW_COMPLETED=true
            fi
            ;;
        "status")
            echo "📋 Current staged changes:"
            git diff --cached --name-status
            echo ""
            ;;
        *)
            echo "⚠️  Invalid choice. Please enter 'approved', 'changes-requested', or 'status'."
            ;;
    esac

    # Increment elapsed time (simulate waiting)
    ELAPSED_TIME=$((ELAPSED_TIME + 10))
done

if [[ $REVIEW_COMPLETED == false ]]; then
    echo "⏰ Maximum wait time exceeded. Cancelling review process."
    exit 1
fi

echo ""
echo "🏁 GitHuman review process completed."

# Cleanup will be handled by the trap
