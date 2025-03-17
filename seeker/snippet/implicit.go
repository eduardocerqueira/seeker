//date: 2025-03-17T16:57:21Z
//url: https://api.github.com/gists/92f6f84d293d63efb3a601e6cede9118
//owner: https://api.github.com/users/shinosaki

package oauth2

import (
	"context"
	"crypto/rand"
	"errors"
	"log"
	"net/http"
	"net/url"
	"strconv"

	"golang.org/x/oauth2"
)

type Implicit struct {
	ctx    context.Context
	cancel context.CancelFunc

	server       *http.Server
	callbackAddr string

	oauth2 *oauth2.Config
	token  *oauth2.Token
	state  string
}

func NewImplicit(clientId string, callbackAddr string, parent context.Context) *Implicit {
	ctx, cancel := context.WithCancel(parent)
	return &Implicit{
		ctx:          ctx,
		cancel:       cancel,
		callbackAddr: callbackAddr,
		oauth2: &oauth2.Config{
			ClientID: clientId,
			Endpoint: oauth2.Endpoint{
				AuthURL:  "https://example.com/oauth2/authorize",
				TokenURL: "**********"://example.com/oauth2/access_token",
			},
		},
	}
}

func (c *Implicit) GetAuthzUrl() *url.URL {
	c.state = rand.Text()
	url, err := url.Parse(c.oauth2.AuthCodeURL(c.state,
		oauth2.SetAuthURLParam("response_type", "token"),
	))
	if err != nil {
		panic(err)
	}
	return url
}

func (c *Implicit) ParseCallbackUrl(query url.Values) (*oauth2.Token, error) {
	if query.Get("state") != c.state {
		return nil, errors.New("invalid state")
	}

	expiresIn, err := strconv.ParseInt(query.Get("expires_in"), 10, 64)
	if err != nil {
		return nil, err
	}

	return &oauth2.Token{
		AccessToken: "**********"
		TokenType: "**********"
		ExpiresIn:   expiresIn,
	}, nil
}

func (c *Implicit) callbackServer() {
	c.server = &http.Server{
		Addr:    c.callbackAddr,
		Handler: nil,
	}

	http.HandleFunc("/callback", func(w http.ResponseWriter, r *http.Request) {
		query := r.URL.Query()
		w.Header().Set("content-type", "text/html")

		if len(query) == 0 {
			log.Println("write")
			w.Write([]byte(`
				<script>
					const { hash } = window.location
					if (hash) {
						window.location.search = hash.substring(1)
					}
				</script>
			`))
		} else {
			token, err : "**********"
			if err != nil {
				http.Error(w, "Parse Callback URL Error:"+err.Error(), http.StatusBadRequest)
			}
			w.Write([]byte(`Authorization successful. Please close this page.`))
			c.token = "**********"
		}
	})

	if err := c.server.ListenAndServe(); err != http.ErrServerClosed {
		panic(err)
	}
}

func (c *Implicit) WaitToken() (*oauth2.Token, error) {
	go c.callbackServer()

	done := make(chan struct{})
	go func() {
		defer close(done)

		for {
			if c.token != "**********"
				return
			}
		}
	}()

	select {
	case <-c.ctx.Done():
		c.server.Close()
		return nil, c.ctx.Err()
	case <-done:
		c.server.Close()
		return c.token, nil
	}
}
urn c.token, nil
	}
}
