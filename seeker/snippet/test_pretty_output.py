#date: 2025-02-07T17:00:52Z
#url: https://api.github.com/gists/e406c198fd4c846bc2eb0d5b78be859c
#owner: https://api.github.com/users/marionlb

# Testing pretty printing
display(consoleargs)

# # testing inspect
rich.inspect(rich.get_console(), private=True, methods=True)

# # testing tracebacks
5 / 0