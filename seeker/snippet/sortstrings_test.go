//date: 2022-02-17T16:52:17Z
//url: https://api.github.com/gists/8a9224f8b2838a9c1d112acde5bdd3c8
//owner: https://api.github.com/users/Deleplace

package bench

import (
	"math/rand"
	"sort"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"
)

const M = 10_000

var data = make([]string, M)

func init() {
	for j := range data {
		data[j] = randomString(12)
	}

}

func BenchmarkSort(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		rand.Shuffle(len(data), func(ii, jj int) { data[ii], data[jj] = data[jj], data[ii] })
		b.StartTimer()

		sort.Strings(data)
	}
}

func BenchmarkSortLess(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		rand.Shuffle(len(data), func(ii, jj int) { data[ii], data[jj] = data[jj], data[ii] })
		b.StartTimer()

		sort.Slice(data, func(ii, jj int) bool { return less(data[ii], data[jj]) })
	}
}

func BenchmarkSortToLower(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		rand.Shuffle(len(data), func(ii, jj int) { data[ii], data[jj] = data[jj], data[ii] })
		b.StartTimer()

		sort.Slice(data, func(ii, jj int) bool { return strings.ToLower(data[ii]) < strings.ToLower(data[jj]) })
	}
}

func BenchmarkSortInsensitiveOptimized(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		rand.Shuffle(len(data), func(ii, jj int) { data[ii], data[jj] = data[jj], data[ii] })
		b.StartTimer()

		sort.Slice(data, func(ii, jj int) bool { return lessCaseInsensitive(data[ii], data[jj]) })
	}
}

// less compares without allocating. It is equivalent to s<t .
func less(s, t string) bool {
	for {
		if len(t) == 0 {
			return false
		}
		if len(s) == 0 {
			return true
		}
		c, sizec := utf8.DecodeRuneInString(s)
		d, sized := utf8.DecodeRuneInString(t)

		if c < d {
			return true
		}
		if c > d {
			return false
		}

		s = s[sizec:]
		t = t[sized:]
	}
}

// lessCaseInsensitive compares s, t without allocating
func lessCaseInsensitive(s, t string) bool {
	for {
		if len(t) == 0 {
			return false
		}
		if len(s) == 0 {
			return true
		}
		c, sizec := utf8.DecodeRuneInString(s)
		d, sized := utf8.DecodeRuneInString(t)

		lowerc := unicode.ToLower(c)
		lowerd := unicode.ToLower(d)

		if lowerc < lowerd {
			return true
		}
		if lowerc > lowerd {
			return false
		}

		s = s[sizec:]
		t = t[sized:]
	}
}

func TestLessCaseInsensitive(t *testing.T) {
	test := t
	for i := 0; i < 1_000_000; i++ {
		s := randomString(12 + i%4)
		t := randomString(12 + i%5)

		less1 := strings.ToLower(s) < strings.ToLower(t)
		less2 := lessCaseInsensitive(s, t)
		if less1 != less2 {
			test.Fatalf("Case-insensitive %q < %q : %v, %v", s, t, less1, less2)
		}
	}
}

const (
	minChar  = 'A'
	maxChar  = 'z'
	spanChar = maxChar - minChar
)

// e.g. "FbpXH\fgTAvx"
func randomString(n int) string {
	buf := make([]byte, n)
	for i := range buf {
		buf[i] = minChar + byte(rand.Intn(spanChar))
	}
	return string(buf)
}
