//date: 2023-10-24T16:34:28Z
//url: https://api.github.com/gists/45b2df58de8271d1ecc22fb97034fb40
//owner: https://api.github.com/users/zacharysyoung

package main

import (
	"fmt"
	"slices"
	"strings"
)

// <https://codereview.stackexchange.com/questions/229042/find-neighboring-pins-on-a-numeric-keypad>
// Your colleague forgot the pin code from the door to the office.
// The keyboard is:
//
//             ┌───┬───┬───┐
//             │ 1 │ 2 │ 3 │
//             ├───┼───┼───┤
//             │ 4 │ 5 │ 6 │
//             ├───┼───┼───┤
//             │ 7 │ 8 │ 9 │
//             └───┼───┼───┘
//                 │ 0 │
//                 └───┘
//
// A colleague recalls that it seems that the pin code was 1357, but
// perhaps each digit should be shifted horizontally or vertically, but
// not diagonally.
//
// For example, instead of 1 it can be 2 or 4, and instead of 5 it can be
// 2, 4, 6 or 8.
//
// Help a colleague get into the office.

var possibles = map[string][]string{
	"1": {"1", "2", "4"},
	"2": {"1", "2", "3", "5"},
	"3": {"2", "3", "6"},
	"4": {"1", "4", "5", "7"},
	"5": {"2", "4", "5", "6", "8"},
	"6": {"3", "5", "6", "9"},
	"7": {"4", "7", "8"},
	"8": {"5", "7", "8", "9", "0"},
	"9": {"6", "8", "9"},
	"0": {"0", "8"},
}

func main() {
	fmt.Println(product(getGroups("46")))   // [13 15 16 19 43 45 46 49 53 55 56 59 73 75 76 79]
	fmt.Println(product(getGroups("1357"))) // [1224 1227 1228 1244 1247 1248 1254 1257 1258 1264 ...]
}

func getGroups(code string) [][]string {
	groups := make([][]string, 0)

	for _, x := range strings.Split(code, "") {
		groups = append(groups, possibles[x])
	}

	return groups
}

func recurse(groups [][]string) []string {
	results := permutate("", groups, len(groups), make([]string, 0))
	slices.Sort(results)
	return results
}

func permutate(s string, groups [][]string, _len int, result []string) []string {
	if len(s) == _len {
		result = append(result, s)
		return result
	}

	for i, subg := range groups {
		for _, y := range subg {
			result = permutate(s+y, groups[i+1:], _len, result)
		}
	}

	return result
}

// a rather clunky translation from
// [Python's itertool.product](https://docs.python.org/3/library/itertools.html#itertools.product)
func product(pools [][]string) []string {
	if len(pools) == 0 {
		return pools[0]
	}

	slots := 1
	for _, pool := range pools {
		slots *= len(pool)
	}
	result := make([]string, slots)
	intermediate := make([]string, slots)

	m := 1
	for _, pool := range pools {
		i := 0
		for _, x := range result[:m] {
			for _, y := range pool {
				intermediate[i] = x + y
				i++
			}
		}
		copy(result, intermediate)
		m *= len(pool)
	}
	return result
}
