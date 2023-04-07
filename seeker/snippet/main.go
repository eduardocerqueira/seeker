//date: 2023-04-07T16:39:52Z
//url: https://api.github.com/gists/8696d6d2545bf9886de3ba6ba00d431d
//owner: https://api.github.com/users/HanEmile

package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

func request(id int, url string) {

	resp, err := http.Get(url)
	if err != nil {
		fmt.Println("Some error: ", err)
	}

	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("Some error: ", err)
	}

	if len(body) != 146 {
		fmt.Printf("[%d] made an http request to url %s | len(req) = %d\n", id, url, len(body))
	}
}

func worker(id int, wordChan chan string, doneChan chan bool) {
out:
	for {
		select {
		case url := <-wordChan:
			url = fmt.Sprintf("https://emile.space/%s", url)
			request(id, url)
		case <-time.After(3 * time.Second):
			fmt.Printf("worker %d couldn't get a new url after 3 seconds, quitting\n", id)
			break out
		}
	}
	doneChan <- true
}

func main() {
	dat, err := os.ReadFile("./wordlist.txt")
	if err != nil {
		fmt.Println("Some error: ", err)
	}

	words := strings.Split(string(dat), "\n")

	wordChan := make(chan string, 10)
	doneChan := make(chan bool, 4)
	for i := 1; i < 4; i++ {
		go worker(i, wordChan, doneChan)
	}

	for _, word := range words {
		wordChan <- word
	}

	for i := 1; i < 4; i++ {
		<-doneChan
	}
}
