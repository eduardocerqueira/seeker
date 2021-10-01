//date: 2021-10-01T01:35:57Z
//url: https://api.github.com/gists/2ad509a19bdf7a4d80d089648177d89e
//owner: https://api.github.com/users/david-yappeter

package main

import "fmt"

var globalVar = 10 // 1st

func init() { // 2nd
	fmt.Println("")
	fmt.Println("==========================")
	fmt.Println("1st init()")
	fmt.Println("==========================")
	fmt.Println("global Var: ", globalVar)
	globalVar = 5
}

func init() { // 3rd
	fmt.Println("")
	fmt.Println("==========================")
	fmt.Println("3nd init()")
	fmt.Println("==========================")
	fmt.Println("global Var: ", globalVar)
	globalVar = 100
}

func main() { // 5th
	fmt.Println("")
	fmt.Println("==========================")
	fmt.Println("main()")
	fmt.Println("==========================")
	fmt.Println("globalVar: ", globalVar)
}

func init() { //4th
	fmt.Println("")
	fmt.Println("==========================")
	fmt.Println("4nd init()")
	fmt.Println("==========================")
	fmt.Println("global Var: ", globalVar)
	globalVar = 50
}
