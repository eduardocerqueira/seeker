//date: 2021-12-03T17:08:36Z
//url: https://api.github.com/gists/b5618263c06ff5574ddec7fbb5f5824e
//owner: https://api.github.com/users/shakahl

package main

import (
	"fmt"
	"sync"
)

var jobs = map[string]string{
	"Job1": "done",
	"Job2": "error",
	"Job3": "done",
	"Job4": "error",
	"Job5": "done",
}

func main() {
	var wg sync.WaitGroup

	resultsChan := make(chan string, len(jobs))
	errChan := make(chan error, len(jobs))

	for jobName, jobStatus := range jobs {
		wg.Add(1)
		go func(j string, s string) {
			defer wg.Done()

			if s == "done" {
				resultsChan <- fmt.Sprintf("%s status: %s", j, s)
			}

			if s == "error" {
				errChan <- fmt.Errorf("%s status: %s", j, s)
				return
			}
		}(jobName, jobStatus)
	}
	wg.Wait()
	close(errChan) <-- Channels need to be closed
	close(resultsChan)  <-- Channels need to be closed
	
	for i := 0; i < len(jobs); i++ {
		if err := <-errChan; err != nil {
			fmt.Printf("Errors=> %s\n", err)
		}
		if results := <-resultsChan; results != "" {

			fmt.Printf("Results=> %s\n", results)
		}
	}
}

/* out:
      Errors=> Job4 status: error
      Results=> Job3 status: done
      Errors=> Job2 status: error
      Results=> Job5 status: done
      Results=> Job1 status: done
*/
