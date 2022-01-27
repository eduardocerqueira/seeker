#date: 2022-01-27T17:12:13Z
#url: https://api.github.com/gists/494eca55f8d8968dd242a3bdbbe1c2fd
#owner: https://api.github.com/users/tw3

#!/usr/bin/env bash

main() {
  echo "Here's an example:"
  echo ""

  echo "\"C:/Program Files/Python39/python.exe\" c:/bin/git-filter-repo.py --commit-callback '"
  echo "msg = commit.message.decode(\"utf-8\")"
  echo "newmsg = msg.replace(\"TEXT_FROM\", \"TEXT_TO\")"
  echo "commit.message = newmsg.encode(\"utf-8\")"
  echo "' --force --refs pilot..BRANCH_NAME"

  echo ""

  echo "1) Copy that into an editor"
  echo "2) Change TEXT_FROM and TEXT_TO and BRANCH_NAME"
  echo "3) Run it in your shell :)"

  echo ""

  echo "For more details visit: https://stackoverflow.com/a/62458610"
}

main "${@}"
