//date: 2023-01-06T16:49:28Z
//url: https://api.github.com/gists/4e2d233f2b6b0815771d873a60eea91a
//owner: https://api.github.com/users/lbvf50mobile

// Leetcode: 1833. Maximum Ice Cream Bars.
// https://leetcode.com/problems/maximum-ice-cream-bars/
// = = = = = = = = = = = = = =
// Accepted.
// Thanks God, Jesus Christ!
// = = = = = = = = = = = = = =
// Runtime: 413 ms, faster than 23.53% of Go online submissions for Maximum Ice Cream Bars.
// Memory Usage: 9.9 MB, less than 17.65% of Go online submissions for Maximum Ice Cream Bars.
// 2023.01.06 Daily Challenge.
import "sort"
func maxIceCream(costs []int, coins int) int {
  sort.Ints(costs)
  counter := 0
  for _,price := range costs {
    if price <= coins {
      counter += 1
      coins -= price
    } else {
      break
    }
  }
  return counter
}
