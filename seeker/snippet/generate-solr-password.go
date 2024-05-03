//date: 2024-05-03T16:54:53Z
//url: https://api.github.com/gists/cd9b01f38003b2eea3466b51394a01e0
//owner: https://api.github.com/users/csandanov

// https://go.dev/play/p/luLFCOgkos9
package main

import (
	"crypto/sha256"
	"encoding/base64"
	"fmt"
)

func main() {
	pass : "**********"
	salt := "salt"
	hasher := sha256.New()
	hasher.Write([]byte(fmt.Sprintf("%s%s", pass, salt)))
	res := hasher.Sum(nil)
	hasher.Reset()
	hasher.Write(res)
	fmt.Printf("%s %s\n", base64.StdEncoding.EncodeToString(hasher.Sum(nil)), base64.StdEncoding.EncodeToString([]byte(salt)))
}
