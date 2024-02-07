//date: 2024-02-07T17:01:36Z
//url: https://api.github.com/gists/833662ca5a76599d750dc50b4d4e1a8f
//owner: https://api.github.com/users/converge

package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel() // Ensure cancellation even if main exits early

	studioCall(ctx, doStuff)
}

func doStuff(ctx context.Context, wg *sync.WaitGroup) {
	for {
		select {
		case <-ctx.Done():
			fmt.Println("exiting...")
			wg.Done()
			return // Exit the function when context is cancelled
		default:
			fmt.Println("...")
			time.Sleep(time.Second) // Simulate work
		}
	}
}

func studioCall(ctx context.Context, userFunc func(ctx context.Context, wg *sync.WaitGroup)) {
	// Create a wait group to ensure all goroutines finish before exiting
	var wg sync.WaitGroup
	wg.Add(1)
	// Wait for all goroutines to finish before returning
	defer wg.Wait()

	// Call the user function with the context
	userFunc(ctx, &wg)
}
