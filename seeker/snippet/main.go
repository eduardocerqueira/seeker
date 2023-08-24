//date: 2023-08-24T16:56:30Z
//url: https://api.github.com/gists/391978953624cb3ecfbeb8904ce74394
//owner: https://api.github.com/users/shareusefulcode

package main

import (
	"fmt"
	"sync"
	"time"
)

var (
	counter int
	mutex   sync.Mutex
)

func incrementCounter() {
	mutex.Lock()
	counter++
	mutex.Unlock()
}

func main() {
	for i := 0; i < 1000000; i++ {
		go incrementCounter()
	}

  // What would happen if we commented out the following line ? and why ?
	time.Sleep(time.Second)

	fmt.Println("Final counter value:", counter)
}
