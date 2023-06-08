//date: 2023-06-08T17:01:10Z
//url: https://api.github.com/gists/31fd574fcf81eb8b153bf9bae9b38365
//owner: https://api.github.com/users/gerardojimenezform3

package main

import (
	"fmt"
	"reflect"
)

func main() {

	for _, c := range []struct {
		left     []string
		right    []string
		expected []string
	}{
		// base cases
		{
			[]string{"a"},
			[]string{},
			[]string{"a"},
		},
		{
			[]string{},
			[]string{"a"},
			[]string{"a"},
		},

		// swap
		{
			[]string{"a", "b", "c", "z"},
			[]string{"a", "c", "d", "z"},
			[]string{"a", "b", "c", "d", "z"},
		},

		// add
		{
			[]string{"a", "z"},
			[]string{"new", "a", "z"},
			[]string{"new", "a", "z"},
		},
		{
			[]string{"a", "z"},
			[]string{"a", "new", "z"},
			[]string{"a", "new", "z"},
		},
		{
			[]string{"a", "z"},
			[]string{"a", "z", "new"},
			[]string{"a", "z", "new"},
		},

		// remove
		{
			[]string{"new", "a", "z"},
			[]string{"a", "z"},
			[]string{"new", "a", "z"},
		},
		{
			[]string{"a", "new", "z"},
			[]string{"a", "z"},
			[]string{"a", "new", "z"},
		},
		{
			[]string{"a", "z", "new"},
			[]string{"a", "z"},
			[]string{"a", "z", "new"},
		},

		// multiple
		{
			[]string{"a", "b", "c", "z"},
			[]string{"a", "x", "z"},
			[]string{"a", "x", "b", "c", "z"},
		},
		{
			[]string{"a", "x", "z"},
			[]string{"a", "b", "c", "z"},
			[]string{"a", "b", "c", "x", "z"},
		},

		// unmergeable
		{
			[]string{"a", "b"},
			[]string{"b", "a"},
			nil,
		},
	} {
		result, _ := merge(c.left, c.right)
		if reflect.DeepEqual(result, c.expected) {
			fmt.Println(c.left, "U", c.right, "=", result, "PASS")
		} else {
			fmt.Println(c.left, "U", c.right, "=", result, "FAIL, but this is expected:", c.expected)
		}
	}

}

func merge(left, right []string) ([]string, error) {
	result := []string{}

	i := 0

	for _, r := range right {
		if inArray(r, left) {

			if i >= len(left) {
				return nil, fmt.Errorf("field '%s' out of order", r)
			}

			for r != left[i] {
				result = append(result, left[i])
				i++
			}
			i++
		}
		result = append(result, r)
	}

	result = append(result, left[i:]...)

	return result, nil
}

func inArray[T comparable](needle T, haystack []T) bool {
	for _, i := range haystack {
		if i == needle {
			return true
		}
	}

	return false
}
