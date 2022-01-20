//date: 2022-01-20T16:58:57Z
//url: https://api.github.com/gists/1662d644b19d0afc2e95aba89de585f5
//owner: https://api.github.com/users/m-murad

package math

import (
	"testing"
)

func TestSum(t *testing.T) {
	t.Parallel()
	
	tt := []struct {
		name        string
		input       []int
		expectedSum int
	}{
		{
			"2ones",
			[]int{1, 1},
			2,
		},
		{
			"5zeros",
			[]int{0, 0, 0, 0, 0},
			0,
		},
		{
			"1to3",
			[]int{1, 2, 3},
			6,
		},
		{
			"+-1",
			[]int{1, -1},
			0,
		},
	}

	for _, tc := range tt {
		tc := tc

		t.Run(tc.name, func(t *testing.T) {
			actualSum := Sum(tc.input)
			if actualSum != tc.expectedSum {
				t.Fatalf("Test %s failed.\nExpected - %d\nGot - %d", tc.name, tc.expectedSum, actualSum)
			}
		})
	}
}
