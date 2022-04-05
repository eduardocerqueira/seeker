//date: 2022-04-05T16:53:16Z
//url: https://api.github.com/gists/60ad33274b50f34704186d0173b71e97
//owner: https://api.github.com/users/Petrakan

package main

import (  
    "fmt"
)

func main() {  
    for i := 0; i < 3; i++ {
        for j := 1; j < 4; j++ {
            fmt.Printf("i = %d , j = %d\n", i, j)
        }
    }
}