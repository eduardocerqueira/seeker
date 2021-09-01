//date: 2021-09-01T01:50:09Z
//url: https://api.github.com/gists/8d5cccec38c5c95e96eb447b28b5391f
//owner: https://api.github.com/users/mrclfd

package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
)

func main() {
	client := &http.Client{}
	var data = []byte(`{url=https%3A%2F%2Fne.com}`)
	req, err := http.NewRequest("POST", "https://s.id/api/public/link/shorten", data)
	if err != nil {
		log.Fatal(err)
	}
	req.Header.Set("User-Agent", "Chrome/62.0 (BSD x86_64; rv:71.0) Gecko/20100101 Firefox/71.0")
	req.Header.Set("Accept", "*/*")
	req.Header.Set("Accept-Language", "en-US,en;q=0.5")
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
	req.Header.Set("X-Requested-With", "XMLHttpRequest")
	req.Header.Set("Origin", "https://s.id")
	req.Header.Set("Connection", "keep-alive")
	req.Header.Set("Referer", "https://s.id/")
	resp, err := client.Do(req)
	if err != nil {
		log.Fatal(err)
	}
	bodyText, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", bodyText)
}
