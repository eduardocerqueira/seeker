//date: 2021-12-03T17:08:36Z
//url: https://api.github.com/gists/b5618263c06ff5574ddec7fbb5f5824e
//owner: https://api.github.com/users/shakahl

package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

type Foo struct {
	sum int
}

func main() {
	var wg sync.WaitGroup
	var mutex = &sync.Mutex{}

	bar := Foo{sum: 0}

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			time.Sleep(time.Duration(rand.Intn(5)) * time.Second)
			mutex.Lock() // <-- Lock sum from being written by more than on goroutine at the same time
			bar.sum++
			mutex.Unlock() // <-- Unlocks sum so it can be written by other goroutine

		}()
	}
	wg.Wait()

	fmt.Printf("Total sum: %d", bar.sum)
}
/* out:
      Total sum: 10 <-- using the mutex lock/unlock no two goroutines can add to sum at the same time
                        creating a race condition
*/
   