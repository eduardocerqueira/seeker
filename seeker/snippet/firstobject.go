//date: 2022-05-24T17:00:51Z
//url: https://api.github.com/gists/3fc3a8642f793f521044406f23cf94a8
//owner: https://api.github.com/users/llimllib

package main

import (
	"bufio"
	"io"
	"os"
)

func main() {
	r := bufio.NewReader(os.Stdin)

	c, err := r.ReadByte()
	if err != nil {
		if err == io.EOF {
			os.Stdout.WriteString("Empty input\n")
			return
		} else {
			panic(err)
		}
	}

	braces := 0
	state := ""
	out := make([]byte, 1024*1024) // set default bufsize to taste
	for {
		if state == "" && c == '{' {
			braces += 1
		} else if state == "" && c == '}' {
			braces -= 1
			if braces == 0 {
				out = append(out, c)
				break
			}
		} else if c == '"' {
			if state == "QUOTED" {
				state = ""
			} else {
				state = "QUOTED"
			}
		} else if state == "QUOTED" && c == '\\' {
			state = "ESCAPE"
		} else if state == "ESCAPE" {
			state = "QUOTED"
		}

		out = append(out, c)

		c, err = r.ReadByte()
		if err != nil {
			if err == io.EOF {
				break
			} else {
				panic(err)
			}
		}

	}

	out = append(out, '\n')
	os.Stdout.Write(out)
}
