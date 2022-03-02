//date: 2022-03-02T17:00:28Z
//url: https://api.github.com/gists/182eeba4b53010f0369a17af872e5644
//owner: https://api.github.com/users/rcanutofelix

package main

import (
	"fmt"
)

func main() {
  var diaDaSemana int
  fmt.Println("Digite o dia da Semana: ")
  fmt.Println("1-Domingo")
  fmt.Println("2-Segunda")
  fmt.Println("3-Terça-feira")
  fmt.Println("4-Quarta-feira")
  fmt.Println("5-Quinta-feira")
  fmt.Println("6-Sexta-feira")
  fmt.Println("7-Sabado")
  
  fmt.Scan(&diaDaSemana)
  
  if diaDaSemana == 1 {
    fmt.Println("Domingo") 
  } else if diaDaSemana == 2 {
    fmt.Println("Segunda")
  } else if diaDaSemana == 3 {
    fmt.Println("Terça-feira")
  } else if diaDaSemana == 4 {
    fmt.Println("Quarta-feira")
  } else if diaDaSemana == 5 {
    fmt.Println("Quinta-feira")
  } else if diaDaSemana == 6 {
    fmt.Println("Sexta-feira")
  } else if diaDaSemana == 7 {
    fmt.Println("Sabádo")
  } else if diaDaSemana == 0 {
   fmt.Println("Valor Inválido!")
  }
}