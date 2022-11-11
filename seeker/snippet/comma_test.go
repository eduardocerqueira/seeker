//date: 2022-11-11T17:08:22Z
//url: https://api.github.com/gists/3a7e0939a243b19adb7af57aeca89710
//owner: https://api.github.com/users/mdwhatcott

package humanize

import "testing"

func TestCommas(t *testing.T) {
	assertEqual(t, "1", Comma(1))
	assertEqual(t, "10", Comma(10))
	assertEqual(t, "100", Comma(100))
	assertEqual(t, "1,000", Comma(1000))
	assertEqual(t, "10,000", Comma(10000))
	assertEqual(t, "100,000", Comma(100000))
}
func assertEqual(t *testing.T, expected, actual string) {
	if actual == expected {
		return
	}
	t.Helper()
	t.Errorf("\n"+
		"want [%s]"+"\n"+
		"got  [%s]", actual, expected)
}
