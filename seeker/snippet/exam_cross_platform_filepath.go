//date: 2022-01-14T16:54:08Z
//url: https://api.github.com/gists/d3cd91ff78cd0577e4999063255e3b22
//owner: https://api.github.com/users/euriion

package main

import (
	"fmt"
	"path/filepath"
)

func main() {
	p := filepath.FromSlash("path/to/file")
	fmt.Println("Path: " + p)
}
