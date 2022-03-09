//date: 2022-03-09T17:13:26Z
//url: https://api.github.com/gists/b4cf2f947ba00f60b0c6e78ef818db16
//owner: https://api.github.com/users/rcanutofelix

package main

import (
	"fmt"
)

func main() {

  var i, numero, maior int

  for i = 1; i <= 5; i++ {

    fmt.Print("Numero ", i, ": ")
    fmt.Scan(&numero)

    if i == 1 {
      maior = numero  
    }
    
    if numero > maior {
      maior = numero
    }
    
  } 
  
  fmt.Println("Maior é ", maior)

}
