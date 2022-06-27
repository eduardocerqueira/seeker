//date: 2022-06-27T17:10:39Z
//url: https://api.github.com/gists/602e8647a737d602c1986aa4137e5327
//owner: https://api.github.com/users/Jibaru

package main

import (
	"fmt"
	"math"
)

type ErrNegativeSqrt float64

func (e ErrNegativeSqrt) Error() string {
	return fmt.Sprintf("cannot Sqrt negative number: %v", float64(e))
}

func Sqrt(x float64) (float64, error) {
	if x < 0 {
		return 0, ErrNegativeSqrt(x)
	}
	z := x/2
	oldZ := -z
	for math.Abs(oldZ - z) > 0.01 {
		oldZ = z
		z -= (z * z - x) / (2 * z)
	}
	return z, nil
}

func main() {
	fmt.Println(Sqrt(2))
	fmt.Println(Sqrt(-2))
}