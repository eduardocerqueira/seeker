//date: 2023-02-07T17:01:28Z
//url: https://api.github.com/gists/f0e0463db5baa268ddd1122b913ebc0d
//owner: https://api.github.com/users/snghnaveen

package main

import (
	"errors"
	"fmt"
	"time"
)

func abandonedReceiver(ch chan int) {
	time.Sleep(time.Hour)
	data := 3

	ch <- data
}

func handler() error {
	ch := make(chan int)
	go abandonedReceiver(ch)

	// if not metioned, then go leak
	defer close(ch)

	if err := dbCall(); err != nil {
		return err
	}

	data := <-ch
	fmt.Println(data)

	return nil
}

func dbCall() error {
	return errors.New("test error")
}

func main() {
	handler()
}
