//date: 2024-01-11T17:05:38Z
//url: https://api.github.com/gists/6a53de56e99111e505310326f79bb506
//owner: https://api.github.com/users/antonxo

package main

import (
	"context"
	"fmt"
	"google.golang.org/api/idtoken"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/", handleMain)
	http.HandleFunc("/tokenSignIn", handleTokenSignIn)
	fmt.Println("Server started at http://localhost:8080")
	http.ListenAndServe(":8080", nil)
}

func handleMain(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "index.html")
}

// Verify Google ID token: "**********"://developers.google.com/identity/gsi/web/guides/verify-google-id-token
func handleTokenSignIn(_ http.ResponseWriter, r *http.Request) {
	// parse POST form
	if err := r.ParseForm(); err != nil {
		log.Fatalln(err)
	}

	// verify CSRF tokens
	const csrfCookieKey = "**********"
	cookieToken, err : "**********"
	if err != nil {
		log.Fatalf("failed to retrieve %s from cookie: %s", csrfCookieKey, err)
	}
	formCookie := r.PostForm.Get(csrfCookieKey)
	if formCookie == "" {
		log.Fatalf("no %s in POST form", csrfCookieKey)
	}
	if cookieToken.Value != "**********"
		log.Fatalln("failed to verify double submit cookie")
	}

	// validate Google token using their library
	payload, err : "**********"
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Print(payload.Claims)
}gleClientId)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Print(payload.Claims)
}