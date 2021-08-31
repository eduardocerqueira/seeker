//date: 2021-08-31T03:19:44Z
//url: https://api.github.com/gists/515f3106c1ca88e9f63b2a50b3c30989
//owner: https://api.github.com/users/marklauyq

package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

func main() {
	maxRoutines := 4
	chn := make(chan int, maxRoutines)

	ctx, doneFn := context.WithCancel(context.Background())
	wg := sync.WaitGroup{}
	for threads := 0; threads < maxRoutines; threads++ {
		wg.Add(1)
		go func(ctx context.Context, wg *sync.WaitGroup, chn chan int, threadID int) {
			var finalCheck bool
		looper:
			for {
				select {
				case <-ctx.Done():
					finalCheck = true
				default:
				}

				processId := <-chn
				fmt.Println("start ", processId, "running on thread :", threadID)
				time.Sleep(time.Second)
				fmt.Println("end", processId)

				if finalCheck {
					break looper
				}
			}
			wg.Done()
		}(ctx, &wg, chn, threads)
	}

	for i := 0; i < 100; i++ {
		chn <- i
	}

	doneFn()
	wg.Wait()
}