//date: 2022-06-24T16:55:13Z
//url: https://api.github.com/gists/fbecbe506cb238e10c3888e7d2720107
//owner: https://api.github.com/users/computerphysicslab

// Sort a histogram of integer keys and values
// https://go.dev/play/p/HJC4k3h_RHo
// Turn
//   map[10:2 11:1 12:2]
// into this:
//   [[10 2] [12 2] [11 1]]
package main

import (
	"fmt"
	"sort"
)

func sortHistogram(m map[int]int) (output [][2]int) {
	n := map[int][]int{} // inverse map value => list of keys
	var a []int          // list of all values

	for k, v := range m {
		n[v] = append(n[v], k) // adding key to the list
	}

	for k := range n {
		a = append(a, k) // populating list of all values
	}
	sort.Sort(sort.Reverse(sort.IntSlice(a))) // values in decrease order

	for _, k := range a { // Looping through values in decrease order
		for _, s := range n[k] { // Looping list of keys
			output = append(output, [2]int{s, k})
		}
	}

	return output // sorted histogram
}

func main() {
	var histogram = make(map[int]int)
	histogram[10]++
	histogram[10]++
	histogram[11]++
	histogram[12]++
	histogram[12]++
	output := sortHistogram(histogram)
	fmt.Printf("\nOutput: %v\n", output)
}