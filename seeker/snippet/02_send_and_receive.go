//date: 2021-10-07T17:08:19Z
//url: https://api.github.com/gists/700b3f6a23f1a185e4d842a2dd5f4956
//owner: https://api.github.com/users/ESidenko

package main

import "fmt"

func main() {
	messages := make(chan string)

	// Send message
	go func() { messages <- "Hello" }()

	// Receive message
	fmt.Println(<-messages) // Hello
}
