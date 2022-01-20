//date: 2022-01-20T16:58:57Z
//url: https://api.github.com/gists/1662d644b19d0afc2e95aba89de585f5
//owner: https://api.github.com/users/m-murad

package math

func Sum(nums []int) int {
	var sum int
	for _, num := range nums {
		sum += num
	}
	return sum
}
