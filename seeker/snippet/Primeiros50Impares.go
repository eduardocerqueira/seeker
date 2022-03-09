//date: 2022-03-09T16:54:20Z
//url: https://api.github.com/gists/7a9a8cd509c6a87fecb3c953124c185c
//owner: https://api.github.com/users/rcanutofelix

package main

import (
	"fmt"
)

func main() {

  var i int

  for i = 1; i <= 50; i++ {

    if i % 2 != 0 {
       fmt.Print(i , ", ")       
    }
    
  } 
 
}
