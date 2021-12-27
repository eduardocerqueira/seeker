//date: 2021-12-27T16:42:38Z
//url: https://api.github.com/gists/f7a360d8d764042cfc5bda0de9f61172
//owner: https://api.github.com/users/dmitruk-v

package main

import (
	"fmt"
	"strings"
)

func main() {
	for i := 0; i < 10; i++ {
		fmt.Printf("%s\n", indent("- ", i))
	}
}

func indent(spacer string, level int) string {
	sb := strings.Builder{}
	for i := 0; i < level; i++ {
		sb.WriteString(spacer)
	}
	return sb.String()
}