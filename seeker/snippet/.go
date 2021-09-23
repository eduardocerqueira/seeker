//date: 2021-09-23T17:15:18Z
//url: https://api.github.com/gists/ce7bbbfaa8cb9555690bc7b5b215c89d
//owner: https://api.github.com/users/rutaka-n

package main

import (
	"fmt"
)

func main() {
	fmt.Println("RESULT:")
	fmt.Println(solution([]int{199, 453, 220, 601}, 6))

	fmt.Println("RESULT:")
	fmt.Println(solution([]int{99, 1}, 100))

	fmt.Println("RESULT:")
	fmt.Println(solution([]int{98, 1}, 100))
}

func solution(accs []int, m int) int {
	total := 0
	for _, acc := range accs {
		total = total + acc
	}
    max := total / m
    min := 1
    for true {
        guess := (max + min) / 2
        if guess == 0 {
            return 0
        }
        if t := tryPay(accs, guess); t != m {
            if t > m {
                min = guess
            } else {
                max = guess
            }

        } else {
            return guess
        }
    }
    return 0
}

func tryPay(accs []int, guess int) int {
    t := 0
    for _, acc := range accs {
        t = t + (acc / guess)
    }
    return t
}
