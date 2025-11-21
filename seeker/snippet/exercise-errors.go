//date: 2025-11-21T16:49:42Z
//url: https://api.github.com/gists/6811a20426a8de2906c2f4ec3e36e773
//owner: https://api.github.com/users/monooso

package main

import (
	"fmt"
	"math"
)

type ErrNegativeSqrt float64

func (err ErrNegativeSqrt) Error() string {
	return fmt.Sprintf("cannot Sqrt negative number: %v", float64(err))
}

func Sqrt(x float64) (float64, error) {
	if x < 0 {
		return 0, ErrNegativeSqrt(x)
	}

	z := 1.0
	diff := 1.0

	for diff > 0.000000000001 {
		z -= (z*z - x) / (2 * z)
		diff = math.Abs((z * z) - x)
	}

	return z, nil
}

func main() {
	fmt.Println(Sqrt(2))
	fmt.Println(Sqrt(-2))
}