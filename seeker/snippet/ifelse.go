//date: 2022-03-22T17:06:47Z
//url: https://api.github.com/gists/af92c63f88f58958228bdf699e2fc4e5
//owner: https://api.github.com/users/ZammaCode

package main

import "fmt"

func main() {
	//declare and initialize variable
	i := 10

	if i < 10 {
		//if i is less than 10 then this
		//block will be executed
		fmt.Println("i < 10  is tue")
	} else if i > 10 && i == 10 {
		//if i is greater than 10 and equal 10
		//then this block will be executed
		fmt.Println("i > 10 && i == 10  is true")
	} else {
		//if neither of the above two conditions are
		//true then this block will be executed
		fmt.Println("Both of the above conditions are false")
	}

	//the composite if allows for initialization
	//with in if statement e.g. here variable j is
	//declared and initialized before test condition
	if j := 2; j == 2 {
		//this block will be executed if j equal to 2
		fmt.Println("j == 2 is true")
	}
}
