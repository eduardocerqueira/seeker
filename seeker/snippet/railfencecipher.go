//date: 2022-01-21T16:47:49Z
//url: https://api.github.com/gists/eec6977f944a25e2612805e529016bcd
//owner: https://api.github.com/users/dhcgn

// Package railfencecipher contains a rail fence cipher challenge
// https://www.codewars.com/kata/58c5577d61aefcf3ff000081/train/go
package railfencecipher

import (
	"math"
	"strings"
)

// getRailNum returns the rail number for the given character index
// This func uses trigonometric functions for fun ;)
// f(x)=0.5×N (1 - cos((π x)/N))
// https://www.wolframalpha.com/input/?i2d=true&i=0.5*3+%5C%2840%291+-+cos%5C%2840%29Divide%5B%5C%2840%29%CF%80+x%5C%2841%29%2C3%5D%5C%2841%29%5C%2841%29
func getRailNum(c int, n int) int {
	railN := 0.5 * float64(n-1) * (1 - math.Cos((math.Pi*float64(c))/float64(n-1)))
	railN = math.Round(railN)
	return int(railN)
}

func Encode(s string, n int) string {
	if s == "" || n < 2 {
		return ""
	}

	// create rails
	rails := make([][]rune, n)
	for i := range rails {
		rails[i] = make([]rune, 0)
	}

	// fill rails
	for c, r := range s {
		n := getRailNum(c, n)
		rails[n] = append(rails[n], r)
	}

	// join rails
	re := ""
	for _, rail := range rails {
		for _, r := range rail {
			re += string(r)
		}
	}

	return re
}

func Decode(s string, n int) string {
	if s == "" || n < 2 {
		return ""
	}

	// create rails
	rails := make([][]rune, n)
	for i := range rails {
		rails[i] = make([]rune, 0)
	}

	// fill rails with dummy runes
	for c, r := range strings.Repeat("~", len(s)) {
		n := getRailNum(c, n)
		rails[n] = append(rails[n], r)
	}

	// fill rails with input string
	c := 0
	for si := 0; c < len(s); si++ {
		for rail_ix := 0; rail_ix < len(rails); rail_ix++ {
			for rune_ix := 0; rune_ix < len(rails[rail_ix]); rune_ix++ {
				rails[rail_ix][rune_ix] = rune(s[c])
				c++
			}
		}
	}

	// deconstrcut rails
	re := ""
	for c := range s {
		n := getRailNum(c, n)

		re += string(rails[int(n)][0])
		rails[int(n)] = rails[int(n)][1:]
	}

	return re
}
