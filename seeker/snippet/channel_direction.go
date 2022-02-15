//date: 2022-02-15T17:04:09Z
//url: https://api.github.com/gists/923a00f3be41ec0f3f532afb6c2458ab
//owner: https://api.github.com/users/tirmizee

package main

import (
	"fmt"
	"time"
)

func main() {
	start := time.Now()

	c := make(chan string)

	defer close(c)

	go sendOnly(c)
	go recieveOnly(c)

	time.Sleep(time.Second * 1)

	end := time.Since(start)
	fmt.Printf("time %v\n", end)

}

func sendOnly(c chan<- string) {
	// invalid operation: cannot receive from send-only channel c
	// data := <-c
	c <- "hello world"
}

func recieveOnly(c <-chan string) {
	// invalid operation: cannot send to receive-only type <-chan string
	// c <- "hello world"
	data := <-c
	fmt.Println(data)
}

// hello world
// time 1.002376684s