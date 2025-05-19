#date: 2025-05-19T16:51:19Z
#url: https://api.github.com/gists/711e610eadf3389fd5cf897b922802a1
#owner: https://api.github.com/users/micahstubbs

# Copyright 2025 Micah Stubbs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

claudep() {
    start_time=$(date +%s)

    PROMPT=$1
    if [ -z "$PROMPT" ]; then
        echo "Usage: claudep <prompt>"
        return 1
    fi
    # Check if claude is installed
    if ! command -v claude &> /dev/null; then
        echo "Claude CLI is not installed. Please install it first."
        return 1
    fi
    # Run the claude command with the provided prompt
    claude -p $PROMPT --output-format stream-json --verbose --allowedTools="Task,Bash,Glob,Grep,LS,Read,Edit,MultiEdit,Write,NotebookRead,NotebookEdit,WebFetch,Batch,TodoRead,TodoWrite,WebSearch,mcp__filesystem__read_file,mcp__filesystem__read_multiple_files,mcp__filesystem__write_file,mcp__filesystem__edit_file,mcp__filesystem__create_directory,mcp__filesystem__list_directory,mcp__filesystem__directory_tree,mcp__filesystem__move_file,mcp__filesystem__search_files,mcp__filesystem__get_file_info,mcp__filesystem__list_allowed_directories,mcp__browser-tools__getConsoleLogs,mcp__browser-tools__getConsoleErrors,mcp__browser-tools__getNetworkErrors,mcp__browser-tools__getNetworkLogs,mcp__browser-tools__takeScreenshot,mcp__browser-tools__getSelectedElement,mcp__browser-tools__wipeLogs,mcp__browser-tools__runAccessibilityAudit,mcp__browser-tools__runPerformanceAudit,mcp__browser-tools__runSEOAudit,mcp__browser-tools__runNextJSAudit,mcp__browser-tools__runDebuggerMode,mcp__browser-tools__runAuditMode,mcp__browser-tools__runBestPracticesAudit,mcp__puppeteer__puppeteer_navigate,mcp__puppeteer__puppeteer_screenshot,mcp__puppeteer__puppeteer_click,mcp__puppeteer__puppeteer_fill,mcp__puppeteer__puppeteer_select,mcp__puppeteer__puppeteer_hover,mcp__puppeteer__puppeteer_evaluate"

    # log out total runtime
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    echo "claude -p command executed with prompt: $PROMPT"
    echo "Total runtime: $runtime seconds, which is $((runtime / 3600)) hours $(( (runtime % 3600) / 60 )) minutes $((runtime % 60)) seconds"

    return 0
}
