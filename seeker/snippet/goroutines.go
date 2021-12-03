//date: 2021-12-03T17:08:36Z
//url: https://api.github.com/gists/b5618263c06ff5574ddec7fbb5f5824e
//owner: https://api.github.com/users/shakahl

package main

import (
	"fmt"
	"sync"
	"time"
)

func main() {
	// the main goroutine **will not wait** for this one to end
	go func() {
		fmt.Println("Start running premature goroutine")
		time.Sleep(10 * time.Minute)
		fmt.Println("End running premature goroutine")
	}()
	fmt.Println("passed prematurely")

	// With a WaitGroup the main goroutine **will** wait for this one to end
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		fmt.Println("Start running waitGroup goroutine")
		time.Sleep(2 * time.Second)
		fmt.Println("End running waitGroup goroutine")
	}()
	wg.Wait()
	fmt.Println("Ended after goroutine execution")
	/* out:
	 	Ended prematurely <-- ignored the goroutine end
		Start running waitGroup goroutine <-- waitgroup goroutine started
		Start running premature goroutine <-- premature goroutine started
		End running waitGroup goroutine <-- waitgroup goroutine endeded
		Ended after waitGroup goroutine execution <-- Waited for the waitGroup but not for the premature goroutine
	
	*/
}
