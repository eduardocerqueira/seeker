//date: 2021-12-20T17:02:57Z
//url: https://api.github.com/gists/c64f44a0fd531b104fb81daaca81cce6
//owner: https://api.github.com/users/m-murad

package math

func Sum(nums []int) int {
	var sum int
	for _, num := range nums {
		sum += num
	}
	return sum
}
