//date: 2022-03-09T17:09:51Z
//url: https://api.github.com/gists/dfb1a7996ae660aa573d5512de369ce5
//owner: https://api.github.com/users/rcanutofelix

package main

import (
	"fmt"
)

func main() {

  var i int
  var numero, soma, media float32

  soma = 0
  
  // soma = soma + numero
  for i = 1; i <= 5; i++ {
    fmt.Print("Numero ", i, ": ")
    fmt.Scan(&numero)
    // soma = soma + numero
    soma += numero 
  } 
  // media = soma/5
  media = soma / 5

  fmt.Println("Soma: ", soma)
  fmt.Println("Media: ", media)

}
