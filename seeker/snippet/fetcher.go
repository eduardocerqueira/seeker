//date: 2022-04-20T17:23:04Z
//url: https://api.github.com/gists/5115995e9ed505e49d56f8a53f00eaa7
//owner: https://api.github.com/users/matino

// Solution to https://go.dev/tour/concurrency/10 with unbuffered channels
package main

import (
	"fmt"
	"sync"
)

type Fetcher interface {
	// Fetch returns the body of URL and a slice of URLs found on that page.
	Fetch(url string) (body string, urls []string, err error)
}

func fetch(fetcher Fetcher, url string) {
	fetched := make(map[string]bool)
	queue := make(chan string, 10)
	var wg sync.WaitGroup

	queue <- url

	// Waits for all goroutines to finish and close the queue channel
	wg.Add(1)
	go func() {
		wg.Wait()
		close(queue)
	}()

	for url := range queue {		
		if _, ok := fetched[url]; ok {
			fmt.Printf("%s already fetched\n", url)
			wg.Done()
			continue
		}
		fetched[url] = true

		fmt.Printf("Processing %s\n", url)

		go func(url string) {
			_, urls, err := fetcher.Fetch(url)
			if err != nil {
				fmt.Printf("Error for %s\n", url)
			} else {
				wg.Add(len(urls))
				for _, url := range urls {
					queue <- url
				}
			}
			wg.Done()
		}(url)
	}
}

func main() {
	fetch(fetcher, "https://golang.org/")
}

// fakeFetcher is Fetcher that returns canned results.
type fakeFetcher map[string]*fakeResult

type fakeResult struct {
	body string
	urls []string
}

func (f fakeFetcher) Fetch(url string) (string, []string, error) {
	if res, ok := f[url]; ok {
		return res.body, res.urls, nil
	}
	return "", nil, fmt.Errorf("not found: %s", url)
}

// fetcher is a populated fakeFetcher.
var fetcher = fakeFetcher{
	"https://golang.org/": &fakeResult{
		"The Go Programming Language",
		[]string{
			"https://golang.org/pkg/",
			"https://golang.org/cmd/",
		},
	},
	"https://golang.org/pkg/": &fakeResult{
		"Packages",
		[]string{
			"https://golang.org/",
			"https://golang.org/cmd/",
			"https://golang.org/pkg/fmt/",
			"https://golang.org/pkg/os/",
		},
	},
	"https://golang.org/pkg/fmt/": &fakeResult{
		"Package fmt",
		[]string{
			"https://golang.org/",
			"https://golang.org/pkg/",
		},
	},
	"https://golang.org/pkg/os/": &fakeResult{
		"Package os",
		[]string{
			"https://golang.org/",
			"https://golang.org/pkg/",
		},
	},
}
