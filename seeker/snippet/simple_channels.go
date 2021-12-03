//date: 2021-12-03T17:08:36Z
//url: https://api.github.com/gists/b5618263c06ff5574ddec7fbb5f5824e
//owner: https://api.github.com/users/shakahl

package main

import (
	"fmt"
	"time"
)

func main() {
	codeChan := make(chan int, 1)
	codeChan <- 2
	fmt.Printf("Initial code: %d\n", <-codeChan)
	go func() {
		time.Sleep(2 * time.Second)
		fmt.Println(" Done after 2 seconds...")
		codeChan <- 3
	}()
  // The channel receiver waits until the chan receives a value to continue
	fmt.Printf("Final code: %d\n", <-codeChan)
	/* out:
	      Initial code: 2
	      Done after 2 seconds...
	      Final code: 3
	*/
}
