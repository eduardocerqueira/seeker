//date: 2021-12-03T17:08:36Z
//url: https://api.github.com/gists/b5618263c06ff5574ddec7fbb5f5824e
//owner: https://api.github.com/users/shakahl

package main

import (
	"fmt"
)

func main() {
	/*
		(1) Unbuffered channel
		For unbuffered channel, the sender will block on the channel until the receiver
		receives the data from the channel, whilst the receiver will also block on the channel
		until sender sends data into the channel.
	*/
	unbufChan := make(chan bool)
	go func(unbufChan chan bool) {
		unbufChan <- true
		unbufChan <- false
		unbufChan <- true
		unbufChan <- true
		close(unbufChan) // <-- The channel needs to be closed to signal it to be unblocked
	}(unbufChan)

	for i := 0; i < 10; i++ {  // <-- The closed channel can still be read from, after all passed values are read it returns the nil value of the chan type
		fmt.Println(<-unbufChan)
	}

	fmt.Println("\n...next example...\n")

	/*
		(2) Buffered channel
		Compared with unbuffered counterpart, the sender of buffered channel will block when
		there is no empty slot of the channel, while the receiver will block on the channel when it is empty.
	*/
	bufChan := make(chan bool, 2)

	bufChan <- true
	bufChan <- false

	for i := 0; i < 2; i++ {

		fmt.Println(<-bufChan)
	}
}
/* out:
true
false
true
true
false
false
false
false
false
false

...next example...

true
false
*/