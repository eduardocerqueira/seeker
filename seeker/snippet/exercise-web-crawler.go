//date: 2023-01-02T16:46:47Z
//url: https://api.github.com/gists/3494099f57a7f9162e292e09ab7036e9
//owner: https://api.github.com/users/teerapap

package main

import (
	"fmt"
	"sync"
)

type Fetcher interface {
	// Fetch returns the body of URL and
	// a slice of URLs found on that page.
	Fetch(url string) (body string, urls []string, err error)
}

type Result struct {
	url   string
	body  string
	links []string
	err   error
}

// Stringer implementation
func (r Result) String() string {
	if r.err != nil {
		return fmt.Sprintf("%v -> error %v", r.url, r.err)
	}
	return fmt.Sprintf("%v -> %v %v", r.url, r.body, r.links)
}

type Crawler struct {
	mu    sync.Mutex
	cache map[string]Result
}

// hasVisisted returns result and true if the crawler has visited the url
func (c *Crawler) hasVisited(url string) (Result, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	res, hit := c.cache[url]
	return res, hit
}

// visit visit the url with the result
func (c *Crawler) visit(url string, res Result) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache[url] = res
}

// Crawl uses fetcher to recursively crawl
// pages starting with url, to a maximum of depth.
func (c *Crawler) Crawl(url string, depth int, fetcher Fetcher, resCh chan Result) {
	defer close(resCh) // close result channel eventually
	if depth <= 0 {
		return
	}

	_, visited := c.hasVisited(url)
	if visited {
		// visited. do not continue
		return
	}

	body, links, err := fetcher.Fetch(url)
	result := Result{url, body, links, err}
	c.visit(url, result)

	// send result back
	resCh <- result
	if err != nil {
		// error result. do not crawl further
		return
	}

	linkResults := make([]chan Result, 0, len(links))
	for _, u := range links {
		ch := make(chan Result)
		linkResults = append(linkResults, ch)
		// spawn to crawl each link in parallel
		go c.Crawl(u, depth-1, fetcher, ch)
	}
	// wait and gather all results from each link
	for _, linkResult := range linkResults {
		for r := range linkResult {
			// return each link result
			resCh <- r
		}
	}
}

func main() {
	results := make(chan Result)
	crawler := &Crawler{cache: make(map[string]Result)}
	go crawler.Crawl("https://golang.org/", 4, fetcher, results)
	for result := range results {
		fmt.Println(result)
	}
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
