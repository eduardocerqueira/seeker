//date: 2021-09-09T17:12:28Z
//url: https://api.github.com/gists/da1248b35db0606e40cdab71d1eb5219
//owner: https://api.github.com/users/TheLunaticScripter

package main

import (
  "bytes"
  ...
  "github.com/open-policy-agent/opa/sdk"
)

func main() {
  ...

  opa, err := sdk.New(ctx, sdk.Options{
		Config: bytes.NewReader(config),
	})
	if err != nil {
		panic(err)
	}

	defer opa.Stop(ctx)
}
