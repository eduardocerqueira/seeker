//date: 2024-03-26T16:53:53Z
//url: https://api.github.com/gists/5946f4baf75ef2aaad920739a469f7bc
//owner: https://api.github.com/users/fabiomartinezmerino

// Our first program will print the classic "hello world"
// message. Here's the full source code.
package main

import (
	"fmt"
	"strconv"
)

func main() {

	//From numeric to string
	var s string = strconv.FormatInt(1000, 10)
	fmt.Println(s)

	var t string = string(1000)
	fmt.Printf("%s\n", t)

	//From string to numeric
	var s_integer, err = strconv.ParseInt(s, 10, 64)
	fmt.Println(s_integer + 3)
	if err != nil {
		fmt.Println(err)
	}

	//Itoa - función que no devuelve error
	scontN := strconv.Itoa(-354)

	fmt.Printf("La cadena contiene este valor %s\n", scontN)

	intFstr, _ := strconv.Atoi(scontN)

	fmt.Printf("Sumamos +1 al valor sacado de una cadena de texto: %d\n", intFstr+1)

}
