//date: 2021-12-03T17:08:36Z
//url: https://api.github.com/gists/b5618263c06ff5574ddec7fbb5f5824e
//owner: https://api.github.com/users/shakahl

package main

import (
	"context"
	"fmt"
	"sync"
)

var jobs = map[string]string{
	"Job1": "done",
	"Job2": "done",
	"Job3": "done",
	"Job4": "error", // <-- this job will always give an "error"
	"Job5": "done",
}

func main() {
	var wg sync.WaitGroup
	ctx, cancel := context.WithCancel(context.Background())

	resultsChan := make(chan string, len(jobs)) // <-- buffered channels make it so 
	errChan := make(chan error, 1)

	for jobName, jobStatus := range jobs {
		wg.Add(1)
		go func(j string, s string) {
			defer wg.Done()

			select {
			case <-ctx.Done():
				return
			default:
			}
			if s == "done" {
				resultsChan <- fmt.Sprintf("%s status: %s", j, s)
			}

			if s == "error" {
				errChan <- fmt.Errorf("%s status: %s", j, s)
				cancel()
				return
			}
		}(jobName, jobStatus)
	}
	wg.Wait()

	if ctx.Err() != nil {
		err := <-errChan
		fmt.Printf("Errors=> %s\n", err.Error())
		return
	}

	for i := 0; i < len(jobs); i++ {
    fmt.Printf("Results=> %s\n", <-resultsChan)
	}
}
/* out:
      Errors=> Job4 status: error <-- One worker gives and error and the context is canceled stopping all other workers
*/