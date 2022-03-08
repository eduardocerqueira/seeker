//date: 2022-03-08T17:07:59Z
//url: https://api.github.com/gists/b538573aed0e5be3ed54c64dd1df3958
//owner: https://api.github.com/users/rcanutofelix

package main

import (
	"fmt"

)

func main() {

  var nota float32

  nota = -1
  
  for nota < 0 || nota > 10 {

    fmt.Print("Nota [0-10] ")
    fmt.Scan(&nota)
    
    if nota < 0 || nota > 10 {
      fmt.Println("Nota invalida")  
    }
    
  }

}
