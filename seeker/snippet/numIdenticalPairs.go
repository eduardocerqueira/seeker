//date: 2023-10-03T16:53:38Z
//url: https://api.github.com/gists/c858d655b99396e9021ced987944036c
//owner: https://api.github.com/users/lbvf50mobile

// Leetcode: 1512. Number of Good Pairs.
// https://leetcode.com/problems/number-of-good-pairs/
// = = = = = = = = = = = = = =
// Accepted.
// Thanks God, Jesus Christ!
// = = = = = = = = = = = = = =
// Runtime: 0 ms, faster than 100.00% of Go online submissions for Number of
// Good Pairs.
// Memory Usage: 2 MB, less than 19.68% of Go online submissions for Number of
// Good Pairs.
// 2023.10.03 Daily Challenge.
package main

func numIdenticalPairs(nums []int) int {
	hash :=make(map[int]int)
	ans := 0
	for _,x := range nums {
		ans += hash[x]
		hash[x] += 1
	}
	return ans
}
