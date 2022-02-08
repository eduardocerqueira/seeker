#date: 2022-02-08T17:00:56Z
#url: https://api.github.com/gists/5233fbce8b5bcd2ebdf53a4c7db3c999
#owner: https://api.github.com/users/pmarreck

# helper function
needs () {
  local bin=$1;
  shift;
  command -v $bin > /dev/null 2>&1 || {
    echo "I require $bin but it's not installed or in PATH; $*" 1>&2;
    return 1
  }
}

pman () {
  needs evince provided by evince package; # instructions may depend on distro
  tmpfile=$(mktemp --suffix=.pdf /tmp/$1.XXXXXX);
  man -Tpdf "$@" >> $tmpfile 2>/dev/null;
  evince $tmpfile &
}