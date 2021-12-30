//date: 2021-12-30T17:20:37Z
//url: https://api.github.com/gists/54b470a0406de53615900573b4c586fc
//owner: https://api.github.com/users/manjeettahkur

package main

import (
	"fmt"
)


func test() (x int) {
	defer func() {
		x++
	}()
	x = 1
	return
}


func anotherTest() int {
	var x int
	defer func() {
		x++
	}()
	x = 1
	return x
}

func aThirdTest() (x int) {
	defer func() {
		x++
	}()
	x = 99
	return 72
}


func main() {
	fmt.Println(test())
	fmt.Println(anotherTest())
	fmt.Println(aThirdTest())
}