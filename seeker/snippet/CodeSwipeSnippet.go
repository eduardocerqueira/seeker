//date: 2021-11-15T17:03:21Z
//url: https://api.github.com/gists/8d3e3f5a7f1c1592937fa97d57032522
//owner: https://api.github.com/users/KishanTeeka

package main

import "fmt"

func fact(n int) int {
    if n == 0 {
        return 1
    }
    return n * fact(n-1)
}

func main() {
    fmt.Println(fact(7))

    var fib func(n int) int

    fib = func(n int) int {
        if n < 2 {
            return n
        }
        return fib(n-1) + fib(n-2)

    }

    fmt.Println(fib(7))
}