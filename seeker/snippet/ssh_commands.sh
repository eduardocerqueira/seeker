#date: 2025-11-27T16:53:50Z
#url: https://api.github.com/gists/c2b696d8aaebf728a07621a82873b742
#owner: https://api.github.com/users/xiexiaoy

seq 1 10 | xargs printf 'eloqdoc-cluster-eloqstore-sysbench-server-%02d\n' | xargs -I{} ssh {} 'hostname -I'