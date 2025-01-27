//date: 2025-01-27T16:43:48Z
//url: https://api.github.com/gists/4b28cc2fed5b731f5d1e3f9f50139d1d
//owner: https://api.github.com/users/anyzicky

package main

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/PuerkitoBio/goquery"
)

type Result struct {
	Url   string
	Title string
	Error error
}

func main() {

	result := make(chan Result, 1)
	urls := []string{
		"https://gamemag.ru/",
		"https://goodgame.ru/",
		"https://www.igromania.ru/",
		"https://universalmotors.ru/",
		"https://www.duolingo.com/learn",
		"https://yourbasic.org/",
		"https://www.twitch.tv/",
	}

	for _, url := range urls {
		go getTitleByUrl(url, result)
	}

	for operation := range result {
		if operation.Error != nil {
			fmt.Printf("Url %s error: %s", operation.Url, operation.Error)
		} else {
			fmt.Println("Result = ", operation.Title)
		}

	}

	//close(result)
	//fmt.Scanln()
}

func getTitleByUrl(url string, result chan<- Result) {

	res, err := http.Get(url)
	if err != nil {
		result <- Result{Url: url, Title: "", Error: err}
	}

	defer res.Body.Close()

	if res.StatusCode != 200 {
		result <- Result{Url: url, Title: "", Error: errors.New("Status code != 200")}
	}

	doc, err := goquery.NewDocumentFromReader(res.Body)
	if err != nil {
		result <- Result{Url: url, Title: "", Error: err}
	}

	title := doc.Find("title").Text()

	result <- Result{Url: url, Title: title, Error: nil}

}
