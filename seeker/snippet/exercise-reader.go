//date: 2025-11-21T16:49:42Z
//url: https://api.github.com/gists/6811a20426a8de2906c2f4ec3e36e773
//owner: https://api.github.com/users/monooso

package main

import "golang.org/x/tour/reader"

type MyReader struct{}

func (r MyReader) Read(b []byte) (int, error) {
	for i := range b {
		b[i] = 'A'
	}
	return len(b), nil
}

func main() {
	reader.Validate(MyReader{})
}