//date: 2021-10-07T17:08:19Z
//url: https://api.github.com/gists/700b3f6a23f1a185e4d842a2dd5f4956
//owner: https://api.github.com/users/ESidenko

package main

func main() {
	messages := make(chan string)

	// Do nothing spawned goroutine
	go func() {}()

	// A groutine ( main groutine ) trying to send message to channel
	// But no other groutine runnning
	// And channel has no buffers
	// So it raises deadlock error
	messages <- "I wanna tell you." // fatal error: all goroutines are asleep - deadlock!
}
