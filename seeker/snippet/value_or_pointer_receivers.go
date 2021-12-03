//date: 2021-12-03T17:08:36Z
//url: https://api.github.com/gists/b5618263c06ff5574ddec7fbb5f5824e
//owner: https://api.github.com/users/shakahl

package main

import (
	"fmt"
)

type Car struct {
	name string
}

//SetName set the name of the **copy** of car
func (car Car) SetName(name string) {
	car.name = name
}

type Moto struct {
	name string
}

//SetName set the name on the **pointer receiver** of mo
func (mo *Moto) SetName(name string) {
	mo.name = name
}

func main() {
	alfa := Car{
		name: "Mito",
	}
	// method as a **value receiver**, name will not change on the struct
	alfa.SetName("giulietta")

	honda := Moto{
		name: "Rebel",
	}
	
	// method as a **pointer receiver**, name will change on the struct
	honda.SetName("Neo Cafe")

	fmt.Printf("alfa: %s\nhonda: %s", alfa.name, honda.name)
  /* out:
	alfa: Mito <-- did not change
	honda: Neo Cafe <-- did change
  */
}
