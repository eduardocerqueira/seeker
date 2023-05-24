//date: 2023-05-24T16:53:36Z
//url: https://api.github.com/gists/f3a6b1cdbc07f55939161ff4deab39c7
//owner: https://api.github.com/users/strazzere

package main

import (
	"fmt"
)

func test() []*int {
	ints := []int{1, 2, 3, 4, 5}
	var newInts []*int

	for index, integer := range ints {
		fmt.Printf("%v : %v\n", integer, &integer) // Pass the address of the beginning address
		fmt.Printf("%v : %v\n", ints[index], &ints[index]) // Pass the correct address of element
		newInts = append(newInts, &ints[index])
	}

	return newInts
}

func main() {
	var allInts []*int
	for i := 0; i < 1; i++ {
		ints := test()
		allInts = append(allInts, ints...)
	}

	fmt.Printf("%v\n", allInts)
	for _, integer := range allInts {
		fmt.Printf("%v\n", *integer)
	}
}