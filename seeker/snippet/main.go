//date: 2024-12-26T16:57:20Z
//url: https://api.github.com/gists/3990060a3d3cb3dd5f60ce1aab18e7ab
//owner: https://api.github.com/users/Mzzqq

package main

import (
	"fmt"
	"net/http"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "Hello, Cloud Run!")
	})

	http.HandleFunc("/hello", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "Hello, it's me Reza")
	})
    
	port := "8080"
	fmt.Println("Listening on port" + port)
	http.ListenAndServe(":"+port, nil)
}
