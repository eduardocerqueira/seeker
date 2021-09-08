#date: 2021-09-08T16:56:39Z
#url: https://api.github.com/gists/954ce398767f15e51729fb969b5e8573
#owner: https://api.github.com/users/khoi-truong

#!/usr/bin/env bash

git_latest_tag() {
  git ls-remote --tags --sort="v:refname" $1 | tail -n1 | sed 's/.*\///; s/\^{}//'
}

git_archive() {
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
  if which $SCRIPT_DIR/$1 >/dev/null; then
    echo "$1 has been installed already"
  else
    echo "Install $1"
    VERSION=$(git_latest_tag "https://github.com/$2.git")
    cd $SCRIPT_DIR
    curl -LJO "https://github.com/$2/releases/download/${VERSION}/$3"
    unzip -o $3 -x $4
    rm -f $3
  fi
}

git_archive swiftlint "realm/SwiftLint" "portable_swiftlint.zip" LICENSE
git_archive swiftformat "nicklockwood/SwiftFormat" "swiftformat.zip"

rm -rf $SCRIPT_DIR/__MACOSX
