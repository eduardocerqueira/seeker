//date: 2021-10-07T17:08:19Z
//url: https://api.github.com/gists/700b3f6a23f1a185e4d842a2dd5f4956
//owner: https://api.github.com/users/ESidenko

package main

import "fmt"
import "time"

func main() {
	messages := make(chan string)

	// Receiver
	go func() {
		fmt.Println("Receiver : I am waiting for your message.")
		msg := <-messages
		fmt.Println("Receiver : I got a mail.")
		fmt.Println(msg)
	}()

	// Sender
	time.Sleep(2000 * time.Millisecond)
	messages <- "Message : Do you like go language?"

	// Wait spawned goroutine process
	time.Sleep(1000 * time.Millisecond)
}
