//date: 2022-04-05T16:58:02Z
//url: https://api.github.com/gists/9b9d5e731229099f21ef7fb1f7c7d48f
//owner: https://api.github.com/users/BadAsstronaut

package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"strings"

	"github.com/google/go-querystring/query"
)

type OauthProviderDetails struct {
	ClientId     string
	ClientSecret string
	AuthorizeUrl string
	TokenUrl     string
}

func oauthDetails() OauthProviderDetails {
	return OauthProviderDetails{
		"XXX",
		"XXX",
		"https://authentik.dev.macroscope.cloud/application/o/authorize/",
		"https://authentik.dev.macroscope.cloud/application/o/token/",
	}
}

func loginHandler(w http.ResponseWriter, r *http.Request) {
	oauthDetails := oauthDetails()
	// TODO: Switch to https://pkg.go.dev/net/url#Values.Encode
	type AuthorizeOptions struct {
		ClientId     string `url:"client_id"`
		Scope        string `url:"scope"`
		ResponseType string `url:"response_type"`
		RedirectUri  string `url:"redirect_uri"`
		AccessType   string `url:"access_type"`
	}

	authorizeOptions := AuthorizeOptions{
		oauthDetails.ClientId,
		"openid email",
		"code",
		"http://localhost:8000/code_redirect",
		"offline",
	}

	qs, _ := query.Values(authorizeOptions)
	authorizeUrl := oauthDetails.AuthorizeUrl + "?" + qs.Encode()
	http.Redirect(w, r, authorizeUrl, http.StatusFound)
}

func codeExchange(w http.ResponseWriter, r *http.Request) {
	client := &http.Client{}
	code := r.URL.Query()["code"][0]
	oauthDetails := oauthDetails()

	payloadData := url.Values{}
	payloadData.Set("grant_type", "authorization_code")
	payloadData.Set("code", code)
	payloadData.Set("client_id", oauthDetails.ClientId)
	payloadData.Set("client_secret", oauthDetails.ClientSecret)
	payloadData.Set("redirect_uri", "http://localhost:8000/code_redirect")

	req, err := http.NewRequest(http.MethodPost, oauthDetails.TokenUrl, strings.NewReader(payloadData.Encode()))

	if err != nil {
		log.Fatal(err)
	}

	req.Header.Add("Content-Type", "application/x-www-form-urlencoded")
	res, err := client.Do(req)

	if err != nil {
		log.Fatal(err)
	}

	defer res.Body.Close()
	body, err := ioutil.ReadAll(res.Body)
	log.Println(string(body))
}

func main() {
	http.HandleFunc("/login", loginHandler)
	http.HandleFunc("/code_redirect", codeExchange)
	fmt.Println("Server starting on 8000")
	if err := http.ListenAndServe(":8000", nil); err != nil {
		log.Fatal(err)
	}
}
