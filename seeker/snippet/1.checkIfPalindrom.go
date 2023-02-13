//date: 2023-02-13T17:06:45Z
//url: https://api.github.com/gists/d83deb7b3527b6008e415f2db4daea33
//owner: https://api.github.com/users/mokhlesurr031

package main

import (
	"fmt"
	"strconv"
)

func checkIfPalindrom(num int) bool {
	if num < 0 {
		return false
	}

	numStr := strconv.Itoa(num)
	strLen := len(numStr)

	for i, j := 0, strLen-1; i < j; i, j = i+1, j-1 {
		if numStr[i] != numStr[j] {
			return false
		}
	}
	return true
}

func main() {

	num := 121121
	res := checkIfPalindrom(num)
	fmt.Println("Palindrom: ", res)

}
