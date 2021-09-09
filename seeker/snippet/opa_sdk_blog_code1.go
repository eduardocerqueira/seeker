//date: 2021-09-09T17:10:24Z
//url: https://api.github.com/gists/75be34b8eccd95b391d2ff69b89d91af
//owner: https://api.github.com/users/TheLunaticScripter

package main

import (
  "context"
  "os"
)

func main() {
  // Setup the context for OPA
  ctx := context.Background()

  // Pull in the OPA config file
  config, err := os.Readfile("opa-conf.yaml")
  if err != nil {

    panic(err)
  }
}

