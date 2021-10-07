//date: 2021-10-07T17:08:19Z
//url: https://api.github.com/gists/700b3f6a23f1a185e4d842a2dd5f4956
//owner: https://api.github.com/users/ESidenko

package main

import "fmt"

// Two groutines are running but deadlock happens

// Output example
//
// Try to receive message
// Try to receive message
// fatal error: all goroutines are asleep - deadlock!

func main() {
	messages := make(chan string)

	go func() {
		fmt.Println("Try to receive message") // Printing
		<-messages                            // Blocking
		fmt.Println("Receive message")        // Never reached
	}()

	fmt.Println("Try to receive message") // Printing
	<-messages                            // Blocking
	fmt.Println("Receive message")        // Never reached

}
