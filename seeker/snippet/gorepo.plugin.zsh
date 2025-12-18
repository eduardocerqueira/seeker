#date: 2025-12-18T17:13:42Z
#url: https://api.github.com/gists/3ee1f8711ebe91141f7b6bd931115ecb
#owner: https://api.github.com/users/chuckmeyer

# gorepo - quickly navigate to GitHub repositories with fuzzy matching
  # Oh My Zsh plugin

  # ============================================================================
  # Configuration - Edit this path to match your GitHub directory location
  # ============================================================================
  typeset -g GOREPO_DIR="$HOME/Documents/GitHub"

  # Helper function: Get list of all repositories (cached)
  _gorepo_get_repos() {
      # Use cache if available and fresh (less than 5 seconds old)
      if [[ -n "$_gorepo_cache" ]] && (( EPOCHSECONDS - ${_gorepo_cache_time:-0} < 5 )); then
          printf '%s\n' "${_gorepo_cache[@]}"
          return 0
      fi

      local -a repos=("root")

      [[ -d "$GOREPO_DIR" ]] || return 1

      for dir in "$GOREPO_DIR"/*(/N); do
          repos+=("${dir:t}")
      done

      # Cache the results
      _gorepo_cache=("${repos[@]}")
      _gorepo_cache_time=$EPOCHSECONDS

      # Return repos via stdout
      printf '%s\n' "${repos[@]}"
  }

  # Helper function: Navigate to a repository
  _gorepo_navigate() {
      local repo="$1"

      if [[ "$repo" == "root" ]]; then
          cd "$GOREPO_DIR" && echo "→ $GOREPO_DIR"
      else
          cd "$GOREPO_DIR/$repo" && echo "→ $GOREPO_DIR/$repo"
      fi
  }

  # Main gorepo function
  gorepo() {
      local pattern="$1"

      # Check if GitHub directory exists
      if [[ ! -d "$GOREPO_DIR" ]]; then
          echo "Error: GitHub directory not found at $GOREPO_DIR"
          return 1
      fi

      # Get all repositories as an array
      local -a repos
      repos=(${(f)"$(_gorepo_get_repos)"})

      # If no repos found (only "root" exists)
      if [[ ${#repos[@]} -eq 1 ]]; then
          echo "No repositories found in $GOREPO_DIR"
          return 1
      fi

      # If pattern provided, check for exact match first
      if [[ -n "$pattern" ]] && (( ${repos[(Ie)$pattern]} )); then
          _gorepo_navigate "$pattern"
          return $?
      fi

      # No exact match or no pattern, use fzf for interactive selection
      local selected
      selected=$(printf '%s\n' "${repos[@]}" | fzf \
          ${pattern:+--query="$pattern"} \
          --height=40% \
          --reverse \
          --border \
          --prompt="Select repo > ")

      # Navigate to selected repository if one was chosen
      [[ -n "$selected" ]] && _gorepo_navigate "$selected"
  }

  # Custom widget for tab completion using fzf
  _gorepo_fzf_completion() {
      # Extract pattern if provided
      local tokens= "**********"
      local pattern="${tokens[2]: "**********"

      # Get all repositories as an array (using cache)
      local -a repos
      repos=(${(f)"$(_gorepo_get_repos)"})

      # Check for exact match if pattern is provided
      if [[ -n "$pattern" ]] && (( ${repos[(Ie)$pattern]} )); then
          BUFFER="gorepo $pattern"
          zle accept-line
          return
      fi

      # Signal ZLE that we're about to perform external I/O
      # This is critical - it tells ZLE to clean up and let fzf have full control
      zle -I

      # No exact match, run fzf for selection
      local selected
      selected=$(printf '%s\n' "${repos[@]}" | fzf \
          ${pattern:+--query="$pattern"} \
          --height=40% \
          --reverse \
          --border \
          --prompt="Select repo > ")

      local ret=$?

      # Refresh the line editor
      zle reset-prompt

      # If something was selected, update command line and execute
      if [[ $ret -eq 0 && -n "$selected" ]]; then
          BUFFER="gorepo $selected"
          zle accept-line
      fi
  }

  # Create and register the widget
  zle -N _gorepo_fzf_completion

  # Save what the current Tab widget is (set by fzf or other plugins)
  _gorepo_save_original_tab_widget() {
      typeset -g _gorepo_original_tab_widget="${$(bindkey '^I')##* }"
  }

  # Create a wrapper that decides whether to use fzf or default completion
  _gorepo_tab_wrapper() {
      local tokens= "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"[ "**********"[ "**********"  "**********"$ "**********"{ "**********"# "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"[ "**********"@ "**********"] "**********"} "**********"  "**********"- "**********"g "**********"t "**********"  "**********"0 "**********"  "**********"] "**********"] "**********"  "**********"& "**********"& "**********"  "**********"[ "**********"[ "**********"  "**********"$ "**********"{ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"[ "**********"1 "**********"] "**********"} "**********"  "**********"= "**********"= "**********"  "**********"" "**********"g "**********"o "**********"r "**********"e "**********"p "**********"o "**********"" "**********"  "**********"] "**********"] "**********"; "**********"  "**********"t "**********"h "**********"e "**********"n "**********"
          _gorepo_fzf_completion
      else
          # Call the original widget that was bound before we took over
          zle ${_gorepo_original_tab_widget:-expand-or-complete}
      fi
  }

  # Create the wrapper widget
  zle -N _gorepo_tab_wrapper

  # Hook to run after all plugins are loaded (after .zshrc sources oh-my-zsh.sh)
  _gorepo_late_init() {
      # Save the current tab widget (likely set by fzf plugin)
      _gorepo_save_original_tab_widget
      # Now bind our wrapper
      bindkey '^I' _gorepo_tab_wrapper
  }

  # Use precmd hook to do this after shell is fully initialized
  autoload -Uz add-zsh-hook
  add-zsh-hook precmd _gorepo_late_init

  # Remove the hook after first run so it doesn't run on every prompt
  _gorepo_cleanup_hook() {
      add-zsh-hook -d precmd _gorepo_late_init
      add-zsh-hook -d precmd _gorepo_cleanup_hook
  }
  add-zsh-hook precmd _gorepo_cleanup_hook