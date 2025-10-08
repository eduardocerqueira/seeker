#date: 2025-10-08T16:50:50Z
#url: https://api.github.com/gists/62155d3bf832e54c4856170e7856fc22
#owner: https://api.github.com/users/btk5301

SHELL_RC="$([ -n \"$ZSH_VERSION\" ] && echo ~/.zshrc || echo ~/.bash_profile)"; \
echo 'export MY_VAR="some_value"' >> "$SHELL_RC" && source "$SHELL_RC"
