#date: 2025-07-18T17:17:24Z
#url: https://api.github.com/gists/5d58c0027efd62bda04505597b9f558c
#owner: https://api.github.com/users/benjishults

urlEncode() {
    printf %s "$1" | jq -sRr @uri
}
