//date: 2023-06-08T17:10:26Z
//url: https://api.github.com/gists/e60efb6b0ceead0766bbebe276902b48
//owner: https://api.github.com/users/kahunacohen

package main

import (
	"fmt"
	"io"
	"net/http"
)

func main() {
	response, err := http.Get("https://example.com")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer response.Body.Close()

	// Read the response body
	htmlBytes, err := io.ReadAll(response.Body)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Convert bytes to string
	html := string(htmlBytes)

	// Print the HTML content
	fmt.Println(html)
}
