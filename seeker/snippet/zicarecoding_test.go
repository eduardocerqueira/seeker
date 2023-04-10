//date: 2023-04-10T17:04:00Z
//url: https://api.github.com/gists/cda5a49855a5d586d0fc2fd0a63a2a5d
//owner: https://api.github.com/users/neverbeenthisweeb

package zicarecoding

import (
	"testing"
)

func TestHasSequence(t *testing.T) {
	cases := []struct {
		target []int
		seq    []int
		ok     bool
	}{
		// Test cases
		{
			target: []int{20, 7, 8, 10, 2, 5, 6},
			seq:    []int{7, 8},
			ok:     true,
		},
		{
			target: []int{20, 7, 8, 10, 2, 5, 6},
			seq:    []int{8, 7},
			ok:     false,
		},
		{
			target: []int{20, 7, 8, 10, 2, 5, 6},
			seq:    []int{7, 10},
			ok:     false,
		},

		// New test cases (candidate's initiative)
		{
			// Empty seq
			target: []int{20, 7, 8, 10, 2, 5, 6},
			seq:    []int{},
			ok:     true,
		},
		{
			// Seq exactly matches with target
			target: []int{20, 7, 8, 10, 2, 5, 6},
			seq:    []int{20, 7, 8, 10, 2, 5, 6},
			ok:     true,
		},
		{
			// First element of seq matches with target
			target: []int{20, 7, 8, 10, 2, 5, 6},
			seq:    []int{20},
			ok:     true,
		},
		{
			// Last element of seq matches with target
			target: []int{20, 7, 8, 10, 2, 5, 6},
			seq:    []int{6},
			ok:     true,
		},
		{
			// Empty seq and empty target
			target: []int{},
			seq:    []int{},
			ok:     true,
		},
		{
			// Non-empty seq but empty target
			target: []int{},
			seq:    []int{1},
			ok:     false,
		},
	}

	for _, tc := range cases {
		ok := HasSequence(tc.target, tc.seq)
		if ok != tc.ok {
			t.Errorf("HasSequence(%v, %v): got %v, want %v", tc.target, tc.seq, ok, tc.ok)
		}
	}
}
