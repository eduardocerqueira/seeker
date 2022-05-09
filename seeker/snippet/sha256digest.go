//date: 2022-05-09T16:53:27Z
//url: https://api.github.com/gists/3b998dc1c7ba78190ddced13a777e1a4
//owner: https://api.github.com/users/francesco-racciatti

package main

import (
	"crypto/sha256"
	"fmt"
)

func get_shasum(value string) string {
	data := []byte(value)
	hash := sha256.Sum256(data)
	return fmt.Sprintf("(sha256) \"%x\"", hash)
}

func main() {
	fmt.Println(get_shasum("my_secret_key"))
}
