//date: 2021-09-01T17:06:01Z
//url: https://api.github.com/gists/e5a213a139666863022fb850a45a0d08
//owner: https://api.github.com/users/wolkenheim

package dangling_pointers

import (
	"fmt"
	"time"
)

func danglingPointer() {
	s := "hello world"

	go func(){
		takesReference(&s)
	}()

	fmt.Println("first func is done")
}

func takesReference(s *string) {
	time.Sleep(100 * time.Millisecond)
	fmt.Printf("this is %s", *s)
	fmt.Println("second func is done")
}