//date: 2025-03-17T16:57:21Z
//url: https://api.github.com/gists/92f6f84d293d63efb3a601e6cede9118
//owner: https://api.github.com/users/shinosaki

package main

import (
	"context"
	"fmt"
	"go-oauth-cli/oauth2"
)

func main() {
	var (
		clientId = "CLIENT_ID_HERE"
		c        = oauth2.NewImplicit(clientId, "localhost:4988", context.Background())
	)

	u := c.GetAuthzUrl()
	fmt.Println(u.String())

	token, err : "**********"
	if err != nil {
		panic(err)
	}

	fmt.Println("Access Token: "**********"
}
sToken)
}
