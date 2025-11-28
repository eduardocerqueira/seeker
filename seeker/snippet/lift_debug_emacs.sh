#date: 2025-11-28T16:51:55Z
#url: https://api.github.com/gists/9db185943e0acbd72f4c9bd5375c4b40
#owner: https://api.github.com/users/BabakSamimi

# Create an exe with debug information
objcopy --only-keep-debug /opt/emacslite/bin/emacs-lite /opt/emacslite/bin/emacs-lite.dbg

# Remove debug info from original exe
strip --strip-debug /opt/emacslite/bin/emacs-lite

# Link the debug info to the exe
objcopy --add-gnu-debuglink=/opt/emacslite/bin/emacs-lite.dbg /opt/emacslite/bin/emacs-lite