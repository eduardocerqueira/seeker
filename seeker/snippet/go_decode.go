//date: 2022-04-14T16:56:55Z
//url: https://api.github.com/gists/ece8d8d7188da20ef3437a109d0b0f12
//owner: https://api.github.com/users/annihilatorrrr

package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"log"
)

func main() {
	input := []byte{14, 255, 129, 4, 1, 2, 255, 130, 0, 1, 12, 1, 12, 0, 0, 12, 255, 130, 0, 1, 3, 102, 111, 111, 3, 98, 97, 114}
	buf := bytes.NewBuffer(input)
	dec := gob.NewDecoder(buf)

	m := make(map[string]string)

	if err := dec.Decode(&m); err != nil {
		log.Fatal(err)
	}

	fmt.Println(m["foo"]) // "bar"
}
