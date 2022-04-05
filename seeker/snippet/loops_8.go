//date: 2022-04-05T17:07:27Z
//url: https://api.github.com/gists/0dec387d83d2238a3c2e4103e2544a1b
//owner: https://api.github.com/users/Petrakan

package main

import (  
    "fmt"
)

func main() {  
outer:  
    for i := 0; i < 3; i++ {
        for j := 1; j < 4; j++ {
            fmt.Printf("i = %d , j = %d\n", i, j)
            if i == j {
                break outer
            }
        }
    }
}