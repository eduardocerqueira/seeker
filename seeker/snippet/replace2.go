//date: 2023-11-14T16:39:48Z
//url: https://api.github.com/gists/f9f723ee8c3168d3def5c6eed614151d
//owner: https://api.github.com/users/miekg

package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"
)

func main() {
	flag.Parse()

	img, err := os.ReadFile(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}
	img = bytes.ReplaceAll(img, []byte("[/color]"), []byte{0x1b, 0x5b, 0x30, 0x6d})
	img = bytes.ReplaceAll(img, []byte("[/font]"), nil)
	img = bytes.ReplaceAll(img, []byte("[/size]"), nil)
	img = bytes.ReplaceAll(img, []byte("[size=9px][font=monospace]"), nil)

	buf := &bytes.Buffer{}

	color := 0
	colstart := 0
	for i, c := range img {
		switch color {
		case 0:
			if c == '[' {
				color++
				continue
			}
			color = 0
		case 1:
			if c == 'c' {
				color++
				continue
			}
			buf.Write([]byte{'['})
			color = 0
		case 2:
			if c == 'o' {
				color++
				continue
			}
			buf.Write([]byte("[c"))
			color = 0
		case 3:
			if c == 'l' {
				color++
				continue
			}
			buf.Write([]byte("[co"))
			color = 0
		case 4:
			if c == 'o' {
				color++
				continue
			}
			color = 0
		case 5:
			if c == 'r' {
				color++
				continue
			}
			color = 0
		case 6:
			if c == '=' {
				color++
				continue
			}
			color = 0
		}

		if color == 7 && c == '#' {
			colstart = i + 1
		}
		if color == 7 && c == ']' {
			// parse each of the 3 tupels to decimal
			dec1, _ := strconv.ParseInt(string(img[colstart+0:colstart+2]), 16, 64)
			dec2, _ := strconv.ParseInt(string(img[colstart+2:colstart+4]), 16, 64)
			dec3, _ := strconv.ParseInt(string(img[colstart+4:colstart+6]), 16, 64)
			color = 0

			cmd := exec.Command("printf", fmt.Sprintf("\033[38;2;%d;%d;%dm", dec1, dec2, dec3))
			control, err := cmd.Output()
			if err != nil {
				log.Fatal(err)
			}

			buf.Write(control)
			continue
		}
		if color == 7 {
			// as long as we are in the parsing colors; don't output anything.
			continue
		}

		buf.Write([]byte{c})
	}

	os.WriteFile("out", buf.Bytes(), 0644)
}