//date: 2021-09-01T17:06:01Z
//url: https://api.github.com/gists/e5a213a139666863022fb850a45a0d08
//owner: https://api.github.com/users/wolkenheim

package dangling_pointers

import (
	"testing"
	"time"
)

/**
* trying to create a dangling pointer in Go
* create variable, pass pointer to another func. first func ends while pointer to variable still exists
* is it a problem in Go? No. GC can handle this. Hence: there are no dangling pointers in Go
* in Rust lifetimes are needed to deal with this
* see https://stackoverflow.com/questions/46987513/handling-dangling-pointers-in-go
 */
func TestDangling(t *testing.T){
	danglingPointer()
	time.Sleep(110 * time.Millisecond)
}
