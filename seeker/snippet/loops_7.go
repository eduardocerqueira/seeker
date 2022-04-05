//date: 2022-04-05T16:57:08Z
//url: https://api.github.com/gists/db9e540e88aa40ba8aef9906080ba2db
//owner: https://api.github.com/users/Petrakan

package main

import (  
    "fmt"
)

func main() {  
    for i := 0; i < 3; i++ {
        for j := 1; j < 4; j++ {
            fmt.Printf("i = %d , j = %d\n", i, j)
            if i == j {
                break
            }
        }
    }
}