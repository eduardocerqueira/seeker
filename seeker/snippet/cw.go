//date: 2022-03-02T17:03:07Z
//url: https://api.github.com/gists/1602922dce1fbbf048ded4514fef209f
//owner: https://api.github.com/users/jaxonwang

package main

import (
	"bytes"
	"fmt"
	"io"
	"math/rand"
	"os"
)

type color struct {
	incolor    bool
	ColorCodes [][]byte
	W          io.Writer
	i          int
}

func (self *color) Write(p []byte) (n int, err error) {

	for {
		if !self.incolor {
			_, err = self.W.Write(self.ColorCodes[self.i])
			self.i = (self.i + 1) % len(self.ColorCodes)
			if err != nil {
				return n, err
			}
			self.incolor = true
		}

		i := bytes.IndexByte(p, byte('\n'))
		if i < 0 { // nothing found
			n1, err := self.W.Write(p)
			n += n1
			return n, err
		}
		n1, err := self.W.Write(p[:i+1])
		n += n1
		if err != nil {
			return n, err
		}
		p = p[i+1:]

		_, err = self.W.Write([]byte("\033[0m"))
		if err != nil {
			return n, err
		}
		self.incolor = false

	}
}

func main() {
	c := &color{ColorCodes: [][]byte{
		[]byte("\033[;31m"),
		[]byte("\033[;32m"),
		[]byte("\033[;33m"),
		[]byte("\033[;34m"),
	},
		W: os.Stdout,
	}

	buf := bytes.NewBuffer([]byte{})

	for i := 0; i < 80; i++ {
		ww := rand.Int()%100 + 10
		to := []byte{}
		for j := 0; j < ww; j++ {
			to = append(to, byte('a')+byte(rand.Int()%27))
		}

		fmt.Fprintln(buf, string(to))
	}

	io.Copy(c, buf)
}
