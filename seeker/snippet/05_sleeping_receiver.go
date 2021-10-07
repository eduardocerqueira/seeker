//date: 2021-10-07T17:08:19Z
//url: https://api.github.com/gists/700b3f6a23f1a185e4d842a2dd5f4956
//owner: https://api.github.com/users/ESidenko

package main

import "fmt"
import "time"

func main() {
	messages := make(chan string)

	// If using buffered chanel instead, sending message will not be blocked
	// messages := make(chan string, 1)

	// Sender
	go func() {
		fmt.Println("Sender : Trying send message.")
		messages <- "Ground control to Majar Tom"
		fmt.Println("Sender : I've already sent message.")
	}()

	// Receiver
	time.Sleep(2000 * time.Millisecond)

	// fmt.Println("Receiver : Trying receive message.")
	msg := <-messages

	fmt.Println("Receiver : Receive message - " + msg)

	// Wait spawned goroutine process
	time.Sleep(1000 * time.Millisecond)
}
