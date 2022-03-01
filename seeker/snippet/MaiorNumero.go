//date: 2022-03-01T16:57:43Z
//url: https://api.github.com/gists/8cd3d46f5769e2c084cae39f08fd6f45
//owner: https://api.github.com/users/samueltcsantos

package main

import (
	"fmt"
)

func main() {
	var numero1 int
	var numero2 int
	fmt.Print("Digite o primeiro número: ")
	fmt.Scan(&numero1)
	fmt.Print("Digite o segundo número: ")
	fmt.Scan(&numero2)

  if numero1 > numero2 {
      fmt.Println("O  maior é ", numero1)
  } else {
      fmt.Println("O  maior é ", numero2) 
  }

}