//date: 2022-09-06T17:03:46Z
//url: https://api.github.com/gists/acab39c37451d0ece5b186e6d46d6806
//owner: https://api.github.com/users/Umenezumi

package main

import "fmt"

func main() {

	array := []string{"a1", "b1", "c1", "d1", "e1"}
	fmt.Println(array)      // [a1 b1 c1 d1 e1]
	fmt.Println(len(array)) // 5
	fmt.Println(cap(array)) // 5

	array2 := array
	array2[0] = "f1"

	fmt.Println(array)  // [f1 b1 c1 d1 e1]
	fmt.Println(array2) // [f1 b1 c1 d1 e1]
}
