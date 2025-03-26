//date: 2025-03-26T16:55:34Z
//url: https://api.github.com/gists/a4786eb2dff87b4e62d1d711da77221d
//owner: https://api.github.com/users/ttys3

package ternary

import "testing"

func assertEquals[T comparable](t *testing.T, a T, b T) {
	if a != b {
		t.Errorf("assertEquals failed: %v != %v", a, b)
	}
}

func Test(t *testing.T) {
	assertEquals(t, If(true, Value("true"), Value("false")), "true")
	assertEquals(t, If(false, Value("true"), Value("false")), "false")

	assertEquals(t, If(true, Func(func() string {
		return "true"
	}), Func(func() string {
		t.Error("lazyEvaluate failed: this func should not be called")
		return "false"
	})), "true")

	assertEquals(t, If(false, Func(func() string {
		t.Error("lazyEvaluate failed: this func should not be called")
		return "true"
	}), Func(func() string {
		return "false"
	})), "false")
}
