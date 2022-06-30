//date: 2022-06-30T21:24:25Z
//url: https://api.github.com/gists/31aa4e7a4d6803a94f80febe328d33e3
//owner: https://api.github.com/users/antklim

package main

import(
	"fmt"
	"strings"

	"github.com/antklim/tuples"
)

func main() {
	formatsConf := "h=700,w=350,f=jpeg h=900,w=450,f=png"

	r, err := tuples.NewReader(strings.NewReader(formatsConf))
	if err != nil {
		fmt.Println(err)
	}

	for {
		tuple, err := r.Read()
		if err == io.EOF {
			break
		}

		if err != nil {
			fmt.Println(err)
		}
		fmt.Printf("%v\n", tuple)
	}

	// Output:
	// [700 350 jpeg]
 	// [900 450 png]
}
