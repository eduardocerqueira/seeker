//date: 2021-12-29T16:59:32Z
//url: https://api.github.com/gists/b0bd96e8a01fd98c0c0c90f1a5b32f53
//owner: https://api.github.com/users/IgnacioFalco

package main

import "fmt"

type animal interface {
	respirar()
	caminar()
	crecer()
}

type leon struct {
	edad int
}

func (l leon) respirar() {
	fmt.Println("El leon respira")
}

func (l leon) caminar() {
	fmt.Println("El leon camina")
}

func (l *leon) crecer() {
	fmt.Printf("El leon tiene %d años antes de crecer\n", l.edad)
	l.edad++
}

func main() {
	var alex animal

	/* alex = leon{edad: 5}
	alex.respirar()
	alex.caminar()
	alex.crecer()
	alex.crecer() */

	alex = &leon{edad: 5}
	alex.respirar()
	alex.caminar()
	alex.crecer()
	alex.crecer()

}
