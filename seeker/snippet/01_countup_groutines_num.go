//date: 2021-10-07T17:08:19Z
//url: https://api.github.com/gists/700b3f6a23f1a185e4d842a2dd5f4956
//owner: https://api.github.com/users/ESidenko

package main

import "fmt"
import "runtime"

func main() {
	// Goroutine num includes main processing
	fmt.Println(runtime.NumGoroutine()) // 1

	// Spawn two goroutines
	go func() {}()
	go func() {}()

	// Total three goroutines run
	fmt.Println(runtime.NumGoroutine()) // 3
}
