#date: 2025-05-16T17:12:33Z
#url: https://api.github.com/gists/a8f025f4c804012ef127f4a60dec962b
#owner: https://api.github.com/users/fontka

#!/bin/sh

# USAGE:
# run 'todo' to search for TAGS recursively and open in quickfix
# redirect if you don't want to open VIM e.g. 'todo > cfile', 'todo | cat'

TAGS='TODO|FIXME|NOTE'
RMTABS_REGEX='s/\([0-9]\+:\)[\t| ]*/\1 /'

[ -t 1 ] && (
	vim -q <(fd -t f | xargs grep -nHE "$TAGS" | sed "$RMTABS_REGEX") \
		--cmd 'autocmd VimEnter * copen | bd2' # START WITH QUICKFIX WINDOW ONLY
		# --cmd 'autocmd VimEnter * cw | wincmd b' # STARTS WITH FIRST MATCH BUFFER AND QUICKFIX
) || (
	fd -t f | xargs grep -nHE $TAGS | sed "$RMTABS_REGEX"
)
