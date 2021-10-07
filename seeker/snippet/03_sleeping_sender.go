//date: 2021-10-07T17:08:19Z
//url: https://api.github.com/gists/700b3f6a23f1a185e4d842a2dd5f4956
//owner: https://api.github.com/users/ESidenko

package main

import "fmt"
import "time"

func main() {
	messages := make(chan string)

	// In spawned goroutine
	//
	// Send message to channel
	// But before do it sleep for a while
	go func() {
		time.Sleep(1000 * time.Millisecond)
		messages <- "Hello"
	}()

	// In main goroutine
	//
	// Receive message from channel
	// Message appears after spawned goroutine awaked from sleeping
	// Channel is empty until other goroutine send message
	// So this receiving will be blocked for a while
	fmt.Println(<-messages) // Hello
}
