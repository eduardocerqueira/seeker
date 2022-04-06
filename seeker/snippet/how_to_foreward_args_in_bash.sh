#date: 2022-04-06T17:14:16Z
#url: https://api.github.com/gists/36b5a5e584e1fea724af73de44a1e527
#owner: https://api.github.com/users/JoshCheek

set -- a 'b c' d
ruby -e 'puts %($*),   ARGV.map { "  #{_1.inspect}" }, ""' $*
ruby -e 'puts %("$*"), ARGV.map { "  #{_1.inspect}" }, ""' "$*"
ruby -e 'puts %($@),   ARGV.map { "  #{_1.inspect}" }, ""' $@
ruby -e 'puts %("$@"), ARGV.map { "  #{_1.inspect}" }, ""' "$@"