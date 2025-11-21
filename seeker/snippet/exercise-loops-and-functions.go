//date: 2025-11-21T16:49:42Z
//url: https://api.github.com/gists/6811a20426a8de2906c2f4ec3e36e773
//owner: https://api.github.com/users/monooso

package main

import (
	"fmt"
	"math"
)

func Sqrt(x float64) float64 {
	z := 1.0
	diff := 1.0

	for diff > 0.000000000001 {
		z -= (z*z - x) / (2 * z)
		diff = math.Abs((z * z) - x)
	}

	return z
}

func main() {
	for x := 2.0; x < 100.0; x += 0.1 {
		myResult := Sqrt(x)
		mathResult := math.Sqrt(x)
		difference := math.Abs(myResult - mathResult)
		
		fmt.Println("My result: ", myResult)
		fmt.Println("Math result: ", mathResult)
		fmt.Println("Difference: ", difference)
		fmt.Println("-------------------------")
	}
}