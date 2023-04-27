//date: 2023-04-27T16:49:37Z
//url: https://api.github.com/gists/872550e685ac2c507a4b7429c6e168d6
//owner: https://api.github.com/users/lbvf50mobile

// Leetcode: 319. Bulb Switcher.
// https://leetcode.com/problems/bulb-switcher/
// = = = = = = = = = = = = = =
// Accpeted.
// Thanks God, Jesus Christ!
// = = = = = = = = = = = = = =
// Runtime: 1 ms, faster than 71.43% of Go online submissions for Bulb Switcher.
// Memory Usage: 1.9 MB, less than 100.00% of Go online submissions for Bulb
// Switcher.
// 2023.04.27 Daily Challenge.
package main

import "math"

func bulbSwitch(n int) int {
	// Based on:
	// https://leetcode.com/problems/bulb-switcher/solution/
	return int(math.Sqrt(float64(n)))
}
