//date: 2022-04-19T17:20:51Z
//url: https://api.github.com/gists/2139479a94d2cb3cb1227d7b91c455df
//owner: https://api.github.com/users/shanna

package main

import (
	"context"
	"fmt"
	"net/http"
	"net/url"

	"github.com/coreos/go-oidc/v3/oidc"
	"github.com/grokify/go-pkce"
	"golang.org/x/oauth2"
)

/*
	Minimal Wails OIDC in-window proof of concept.

	In your JS depending on what template you generated:

	let email = await window.go.main.App.AuthEmail();

	if (!email) {
		window.go.main.App.AuthURL().then((url) => {
			document.location.href = url;
		});
	}
*/

// Wails V2 application URI.
const FrontendURI = "wails://wails/"

const OAuth2Issuer = "https://accounts.google.com"

// The Go HTTP endpoint we'll spin up to handle oauth2 redirects.
//
// In a real app you'd want to use an ephemeral port.
var OAuth2RedirectURL = "http://localhost:9991/auth/callback"

type Claims struct {
	Email    string `json:"email"`
	Verified bool   `json:"email_verified"`
}

// App struct
type App struct {
	ctx      context.Context
	provider *oidc.Provider
	auth     *oauth2.Config
	claims   *Claims
	verifier string
}

func NewApp() (*App, error) {
	verifier := pkce.NewCodeVerifier()

	// OIDC though you can just create the URLs yourself.
	provider, err := oidc.NewProvider(context.Background(), OAuth2Issuer)
	if err != nil {
		return nil, fmt.Errorf("oidc issuer: %w", err)
	}

	// I don't get why bug Google seems to require a secret with PKCE still unless I missing something in my
	// implementation?
	//
	// Google generates a secret for web app clients still and I don't seem to be able to omit it or provide a dummy
	// value. Looking around Stack Overflow I see a lot of confusion.
	auth := &oauth2.Config{
		Endpoint:     provider.Endpoint(),
		ClientID:     "client-id",
		ClientSecret: "not-so-secret-secret",
		Scopes:       []string{oidc.ScopeOpenID, "profile", "email"},
		RedirectURL:  OAuth2RedirectURL,
	}

	app := &App{
		provider: provider,
		auth:     auth,
		verifier: verifier, // TODO: Needs storage if you don't want to log in each app-start.
	}

	return app, nil
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx

	// TODO: The handler path needs to be whatever you set up on the OIDC provider.
	http.HandleFunc("/auth/callback", func(w http.ResponseWriter, r *http.Request) {
		// TODO: Emit events for auth errors and success?

		// Check errors on this lot obv:

		token, _ := a.auth.Exchange(
			ctx,
			r.URL.Query().Get("code"),
			oauth2.SetAuthURLParam(pkce.ParamCodeVerifier, a.verifier),
		)

		rawIDToken, _ := token.Extra("id_token").(string)

		idToken, _ := a.provider.Verifier(&oidc.Config{ClientID: a.auth.ClientID}).Verify(ctx, rawIDToken)

		_ = idToken.Claims(&a.claims)

		// Wails needs a runtime.WindowOpenURL(ctx, "url") equivalent of runtime.BrowserOpenURL(ctx, "url") perhaps? If the
		// window has navigated away from wails://wails then there is no easy way to navigate back unless the page the window
		// is displaying happens to be under your control (like this handler). If this is the case you can't 301 to
		// wails://wails but you can in javascript. No idea why.
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "<script>window.location.href=%q</script>", FrontendURI)
	})
	go http.ListenAndServe(":9991", nil)
}

// domReady is called after the front-end dom has been loaded
func (a App) domReady(ctx context.Context) {
	// Add your action here
}

// shutdown is called at application termination
func (a *App) shutdown(ctx context.Context) {
	// Perform your teardown here
}

// Greet returns a greeting for the given name
func (a *App) Greet(name string) string {
	return fmt.Sprintf("Hello %s!", name)
}

// Generate an OAuth2 login URL.
//
// Redirect the Wails window (browser) in the frontend.
func (a *App) AuthURL() string {
	challenge := pkce.CodeChallengeS256(a.verifier)

	return a.auth.AuthCodeURL(
		"state",
		oauth2.SetAuthURLParam(pkce.ParamCodeChallenge, challenge),
		oauth2.SetAuthURLParam(pkce.ParamCodeChallengeMethod, pkce.MethodS256),
  )
}

// If the AuthEmail returns empty string then we haven't logged in yet so redirect to AuthURL.
func (a *App) AuthEmail() string {
	if a.claims != nil {
		return a.claims.Email
	}
	return ""
}