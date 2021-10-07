//date: 2021-10-07T17:08:19Z
//url: https://api.github.com/gists/700b3f6a23f1a185e4d842a2dd5f4956
//owner: https://api.github.com/users/ESidenko

package main

func main() {
	messages := make(chan string)

	// Do nothing spawned goroutine
	go func() {}()

	// A groutine ( main groutine ) trying to receive message from channel
	// But channel has no messages, it is empty.
	// And no other groutine running. ( means no "Sender" exists )
	// So channel will be deadlocking
	<-messages // fatal error: all goroutines are asleep - deadlock!
}
