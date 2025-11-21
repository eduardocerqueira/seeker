//date: 2025-11-21T16:49:42Z
//url: https://api.github.com/gists/6811a20426a8de2906c2f4ec3e36e773
//owner: https://api.github.com/users/monooso

package main

import (
	"io"
	"os"
	"strings"
)

type rot13Reader struct {
	r io.Reader
}

func rotate(b byte) byte {
	switch {
	case b >= 'A' && b <= 'Z':
		r := b + 13
		if r > 'Z' {
			r = r - 26
		}
		return r
	case b >= 'a' && b <= 'z':
		r := b + 13
		if r > 'z' {
			r = r - 26
		}
		return r
	default:
		return b
	}
}

func (rot *rot13Reader) Read(b []byte) (int, error) {
	n, err := rot.r.Read(b)

	if err != nil {
		return n, err
	}

	for i := range b {
		b[i] = rotate(b[i])
	}

	return len(b), nil
}

func main() {
	s := strings.NewReader("Lbh penpxrq gur pbqr!")
	r := rot13Reader{s}
	io.Copy(os.Stdout, &r)
}