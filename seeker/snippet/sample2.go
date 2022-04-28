//date: 2022-04-28T17:04:20Z
//url: https://api.github.com/gists/dafcf99bcb2cf8ab6d2a71dd5620f93c
//owner: https://api.github.com/users/ivancorrales

package main

import (
	"fmt"
	"syscall/js"
	"time"
)

func main() {
	currentTime := time.Now().Format("2006-01-02 15:04:05")
	h2Element := js.Global().Get("document").
		Call("getElementById", "description")
	msg := fmt.Sprintf(
		"Sample 2 | Modifying the HTML (%s)",
		currentTime)
	h2Element.Set("innerHTML", msg)
}
