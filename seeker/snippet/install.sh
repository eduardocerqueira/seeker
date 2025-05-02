#date: 2025-05-02T16:25:28Z
#url: https://api.github.com/gists/cb4ef21925b33254ae718d0ddc0c470d
#owner: https://api.github.com/users/kafai-lam

#!/bin/bash

command_exists() {
    command -v "$1" &> /dev/null
}

echo "This script will install and configure the llm command with Git integration."

if ! command_exists uv; then
    echo "Step 1: Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    if ! command_exists uv; then
        echo "Failed to install uv. Please install manually and run the script again."
        exit 1
    fi
else
    echo "Step 1: uv is already installed."
fi

echo "Step 2: Using uv to install llm with llm-git and llm-mistral..."

uv tool install --with llm-git,llm-mistral llm

echo "Step 3: Create config for custom commit guidelines..."
mkdir -p ~/.config/llm-git
cat > ~/.config/llm-git/config.yaml << 'EOL'
terminal:
  # Theme for syntax highlighting
  theme: "monokai"
  # Style for markdown rendering
  markdown_style: "default"
  # Width for syntax highlighted content (null means use full terminal width)
  syntax_width: null
  # Width for markdown content like commit messages
  markdown_width: 72
  # Color system for terminal output
  color_system: "auto"
  # Whether to use syntax highlighting
  highlight: true

prompts:
  # Prompts can reference earlier prompts using the {prompt[name]} syntax

  assistant_intro: |
    # Git Assistant
    You are a git assistant.
    Line length for text output is 72 characters.
    Use instructions below to guide your responses.
    Later instructions have precedence over earlier ones.

  writing_style: |
    ## Writing Style
    - Use imperative statements in the subject line, e.g. "Fix broken Javadoc link"
    - Begin the subject line sentence with a capitalized verb, e.g. "Add, Prune, Fix, Introduce, Avoid, etc"
    - Do not end the subject line with a period
    - Keep the subject line to 50 characters or less if possible
    - Wrap lines in the body at 72 characters or less
    - Use the body to explain what and why, not how
    - For small changes, 1 or 2 sentences are enough
    - Avoid filler words

  extend_prompt: |
    {old_prompt}

    ## Additional Instructions
    {add_prompt}

  extend_prompt_commit_metadata: |
    {old_prompt}

  # Shared sections for commit messages
  commit_intro: |
    {prompt[assistant_intro]}

    {prompt[writing_style]}

    ## Context
    - Working directory: `{pwd}`
    - Current branch: `{branch}`

  commit_requirements: |
    ## Requirements
    Commit message has four parts: tag, module, short description and full description.

    ### Message Format
    <template>
    [<tag>] <module>: <short description (< 50 chars)>

    <Long description, including the rationale for the change, or a summary of the feature being introduced>

    <Related Ticket Number if mentioned in Additional Instructions or branch name>
    <example: Ticket: TICKET-123, Close: ISSUE-5=456, Issue: JIRA-789>
    </template>

    #### Tags
    - [FIX] for bug fixes: mostly used in stable version but also valid if you are fixing a recent bug in development version
    - [REF] for refactoring: when a feature is heavily rewritten
    - [ADD] for adding new modules
    - [REM] for removing resources: removing dead code, removing views, removing module
    - [REV] for reverting commits: if a commit causes issues or is not wanted reverting it is done using this tag
    - [MOV] for moving files: use git move and do not change content of moved file otherwise Git may loose track and history of the file; also used when moving code from one file to another
    - [REL] for release commits: new major or minor stable versions
    - [IMP] for improvements: most of the changes done in development version are incremental improvements not related to another tag
    - [MERGE] for merge commits: used in forward port of bug fixes but also as main commit for feature involving several separated commits
    - [CLA] for signing the Odoo Individual Contributor License
    - [I18N] for changes in translation files
    - [PERF] for performance patches

    #### Module
    After tag comes the modified module name.
    Use the technical name as functional name may change with time.
    If several modules are modified, list them to tell it is cross-modules.

    ### Examples
    <example>
    [REF] models: use `parent_path` to implement parent_store

    This replaces the former modified preorder tree traversal (MPTT) with
    the fields `parent_left`/`parent_right`
    <example>

    <example>
    [FIX] account: remove frenglish
    </example>

    <example>
    [FIX] website: fixes look of input-group-btn

    Bootstrap's CSS depends on the input-group-btn element being the
    first/last child of its parent. This was not the case because of the
    invisible and useless alert.
    </example>

  # Regular commit message template
  commit_message: |
    {prompt[commit_intro]}

    {prompt[commit_requirements]}

    ## Output
    Only output the commit message.

  # Amend commit message template
  commit_message_amend: |
    {prompt[commit_intro]}
    - You are amending an existing commit

    {prompt[commit_requirements]}
    - Consider both the previous commit message and the new changes
    - Maintain the same scope and type as the original commit if appropriate

    ## Previous Commit Message
    ```
    {previous_message}
    ```

    ## Output
    Only output the updated commit message.

  branch_name: |
    {prompt[assistant_intro]}

    ## Task
    Extract a one line branch name from the commit range.

    ## Format
    Use the following pattern: `TICKET-1234.description-of-the-branch`
    where TICKET-1234 is the ticket number extracted from the commit messages.

    ## Output
    Only output the branch name and nothing else.

  tag_name: |
    {prompt[assistant_intro]}

    ## Task
    Generate a suitable tag name and a concise tag message from the commit range.

    ## Format
    Tag Name: Use semantic versioning (e.g., `v1.2.3`) or a descriptive name (e.g., `release-candidate-feature-x`). Consider the nature of the changes (fix, feature, breaking change) when deciding on the version bump or name.
    Tag Message: Provide a brief summary of the changes included in this tag. Follow commit message conventions (imperative mood, concise).

    ## Output
    Output the tag name on the first line.
    Output the tag message on the subsequent lines, separated from the tag name by a single blank line.
    Example:
    v1.1.0

    feat: Add user authentication

    This release introduces user login and registration functionality.

  pr_description: |
    {prompt[assistant_intro]}

    {prompt[writing_style]}

    ## Task
    Create a pull request description based on the commits in the current branch.

    ## Requirements
    - The PR title is the first line of the description
    - Use conventional commits format for the PR description and PR title
    - Extract a type and scope from the commits to come up with the PR title
    - Extract the ticket number from the tickets and put it in the footer

    ## Output
    Only output the PR description.

  describe_staged: |
    {prompt[assistant_intro]}

    {prompt[writing_style]}

    ## Task
    Describe the changes in the given diff.

    ## Output
    1. Summarize the changes
    2. Suggest ways to split the changes into multiple commits

  split_diff: |
    {prompt[assistant_intro]}

    ## Task
    Split the diff into multiple atomic commits.

    ## Output
    Extract the first commit of the sequence.

  apply_patch_base: |
    {prompt[assistant_intro]}

    ## Output Requirements
    Output a patch that can be applied cleanly with `git apply --cached`.
    It must be relative to HEAD.

  apply_patch_custom_instructions: |
    {prompt[apply_patch_base]}

    ## Instructions
    {instructions}

  apply_patch_minimal: |
    {prompt[apply_patch_base]}

    ## Output Requirements
    - Focus on the most important changes first
    - Prioritize logical groupings of changes
    - Ensure the patch can be applied cleanly
    - Only include changes that make sense together

  improve_rebase_plan: |
    {prompt[assistant_intro]}

    You are an expert Git user helping to improve a rebase plan.

    ## Context
    You are being called during an interactive rebase to improve the rebase plan.

    ## Rewrite commit messages

    Use the instruction
    ```
    exec llm git commit --amend [--extend-prompt "INSTRUCTIONS"]
    ```
    to rewrite commit messages.

    ## Requirements
    - Analyze the rebase plan and commit details
    - Look for opportunities to:
      - Squash related commits.
      - Reorder commits logically
      - Make sure to avoid conflicts. Be conservative.
    - Be hesitant to drop any commits. When dropping a commit, add a comment explaining why.
    - Unless you want to leave a commit message exactly as it is, use the `exec` command to rewrite it.
    - Return ONLY the improved rebase plan, nothing else
    - Maintain the same format as the original plan
    - Each line must start with a command (pick, reword, squash, exec, etc.) followed by the commit hash
    - Do not change the commit hashes
    - Omit the standard instructions that are commented out.

    ## Output
    Return only the improved rebase plan, maintaining the exact format required by Git.

  rebase_input: |
    Rebase plan:
    ```
    {rebase_plan}
    ```

    Commit details:
    ```
    {commit_details}
    ```

git:
  # Files to exclude from diffs and shows
  exclude_files:
    # Package lock files
    - "package-lock.json"
    - "yarn.lock"
    - "pnpm-lock.yaml"
    - "npm-shrinkwrap.json"
    - "bun.lockb"

    # Dependency directories
    - "node_modules/"
    - "vendor/"

    # Generated files
    - "dist/"
    - "build/"

    # Large data files
    - "*.min.js"
    - "*.min.css"

    # Python virtual environments
    - "venv/"
    - ".venv/"
    - "env/"

    # Compiled Python files
    - "__pycache__/"
    - "*.pyc"

    # Generated documentation
    - "docs/_build/"

    # Generated translations
    - "*.mo"

    # Database files
    - "*.sqlite3"
    - "*.db"

    - uv.lock
EOL

echo "Setup complete! Please add your model config."
echo "e.g. llm models default <model_id>"
echo "e.g. llm models options set <model_id> temperature 0.6"