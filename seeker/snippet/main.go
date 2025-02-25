//date: 2025-02-25T16:58:54Z
//url: https://api.github.com/gists/5505f835aea32cf4e84add0e448c4ce0
//owner: https://api.github.com/users/shinderuman

package main

import (
	"context"
	"log"

	"github.com/mattn/go-mastodon"
)

func main() {
	config := &mastodon.Config{
		Server:       "https://kenji.asmodeus.jp",
		ClientID:     "",
		ClientSecret: "**********"
		AccessToken: "**********"
	}
	c := mastodon.NewClient(config)

	wsc := c.NewWSClient()
	q, err := wsc.StreamingWSPublic(context.Background(), true)
	if err != nil {
		log.Fatal(err)
	}
	for e := range q {
		if t, ok := e.(*mastodon.UpdateEvent); ok {
			if t.Status.Account.Username == "kindle_sale" {
				bt, err := c.Reblog(context.Background(), t.Status.ID)
				if err != nil {
					log.Println("ブーストエラー: ", err)
				} else {
					log.Println(bt.URL)
				}
			}
		}
	}
}
