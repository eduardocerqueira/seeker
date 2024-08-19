//date: 2024-08-19T17:09:32Z
//url: https://api.github.com/gists/8884d830d39881dd1897ddbb0053b057
//owner: https://api.github.com/users/davenmurphy

package main

//Number guessing game
import (
	"fmt"
	"math/rand"
)

func main() {

	RandomInteger := rand.Intn(10 - 0)

	fmt.Println("Pick a number between 1 - 10")

	var input int16
	for {
		_, err := fmt.Scanln(&input)
		if err != nil {
			break
		}
	}

	if input == int16(RandomInteger) {
		println("Correct!")
	} else {
		println("Nope")
	}
}
