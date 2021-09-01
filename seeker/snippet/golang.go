//date: 2021-09-01T17:03:17Z
//url: https://api.github.com/gists/9555324ca5d5faf81e80b427086e3878
//owner: https://api.github.com/users/techno-tanoC

package main

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
)

func main() {
}

func notifySlack(url string, message string) error {
	values := map[string]string{
		"text": message,
	}
	data, err := json.Marshal(values)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := http.Post(url, "application/json", bytes.NewReader(data))
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	return err
}
