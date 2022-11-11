//date: 2022-11-11T17:08:22Z
//url: https://api.github.com/gists/3a7e0939a243b19adb7af57aeca89710
//owner: https://api.github.com/users/mdwhatcott

package humanize

import (
	"fmt"
	"strings"
)

// Comma formats a number with commas separating each block of 3 numbers.
// Inspiration: https://pkg.go.dev/github.com/dustin/go-humanize#Comma
func Comma(n int) string {
	s := []rune(Reverse(fmt.Sprint(n)))
	var b strings.Builder
	for x := 0; ; x++ {
		b.WriteRune(s[0])
		s = s[1:]
		if len(s) > 0 && (x+1)%3 == 0 {
			b.WriteRune(',')
		}
		if len(s) == 0 {
			return Reverse(b.String())
		}
	}
}
func Reverse(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}
