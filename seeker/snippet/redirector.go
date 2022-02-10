//date: 2022-02-10T17:02:41Z
//url: https://api.github.com/gists/918d2b9cfcc0dc904594fb44ca53f831
//owner: https://api.github.com/users/candlerb

package main

import (
	"fmt"
	"net/http"
	"os"
	"strings"
)

func redirect(w http.ResponseWriter, req *http.Request) {
	parts := strings.SplitN(req.Host, ":", 2)
	host := parts[0]
	if host == "" {
		w.WriteHeader(400)
		w.Write([]byte("Invalid request, Host header missing"))
		return
	}
	location := "https://" + host + req.URL.Path
	w.Header().Set("Location", location)
	w.WriteHeader(301)
}

func main() {
	http.HandleFunc("/", redirect)
	err := http.ListenAndServe(":80", nil)
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}