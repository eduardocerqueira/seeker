//date: 2022-12-13T17:07:38Z
//url: https://api.github.com/gists/9bf0bc59f0f73af51a740f49fe455c94
//owner: https://api.github.com/users/sakshamsaxena

package main

import (
	"fmt"
)

func main() {
	/*
		Given a stream of numbers,
		use two goroutines to print odd and even numbers by each
		in the input sequence itself.
	*/
	stream := []int{2, 4, 5, 6, 8, 9, 10, 40, 89}

	evenChan := make(chan int, 0) // unbuffered
	oddChan := make(chan int, 0)  // unbuffered
	open := make(chan bool, 1)    // buffer of 1

	go printOddNumbers(oddChan, open)
	go printEvenNumbers(evenChan, open)

	open <- true // mark 1 channel as open whichever gets the first number
	for _, number := range stream {
		if number%2 == 0 {
			evenChan <- number
		} else {
			oddChan <- number
		}
	}
	<-open // last is opened at the end
}

func printEvenNumbers(numbers chan int, open chan bool) {
	for {
		select {
		case num := <-numbers:
			<-open
			fmt.Println(num)
			open <- true
		default:

		}
	}
}

func printOddNumbers(numbers chan int, open chan bool) {
	for {
		select {
		case num := <-numbers:
			<-open
			fmt.Println(num)
			open <- true
		default:

		}
	}
}
