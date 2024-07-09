//date: 2024-07-09T16:53:54Z
//url: https://api.github.com/gists/ec88f23ff4011b784ad92a7406529876
//owner: https://api.github.com/users/elliotcubit

package main

import (
	"bytes"
	"fmt"
)

const Flag = "xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx"
const UUIDv4StringLen = len(Flag)
const FlagStringLen = UUIDv4StringLen + 6

func SearchString(mem *Memory) (string, error) {
	p := 0
	for {
		page, err := mem.ReadPage(p)
		if err != nil {
			return "", err
		}

		// Check for the full prefix
		idx := bytes.Index(page, []byte("gc24{"))
		if idx != -1 {
			s, err := flagAtIdx(mem, p*len(page)+idx)
			if err != nil {
				return "", err
			}

			if s != "" {
				fmt.Println(s)
				return s, nil
			}
		}

		// Or these prefixes, at the end of the page
		idx = bytes.LastIndex(page, []byte("gc24"))
		if idx == len(page)-4 {
			s, err := flagAtIdx(mem, p*len(page)+idx)
			if err != nil {
				return "", err
			}

			if s != "" {
				fmt.Println(s)
				return s, nil
			}
		}

		idx = bytes.LastIndex(page, []byte("gc2"))
		if idx == len(page)-3 {
			s, err := flagAtIdx(mem, p*len(page)+idx)
			if err != nil {
				return "", err
			}

			if s != "" {
				fmt.Println(s)
				return s, nil
			}
		}

		idx = bytes.LastIndex(page, []byte("gc"))
		if idx == len(page)-2 {
			s, err := flagAtIdx(mem, p*len(page)+idx)
			if err != nil {
				return "", err
			}

			if s != "" {
				fmt.Println(s)
				return s, nil
			}
		}

		idx = bytes.LastIndexByte(page, 'g')
		if idx == len(page)-1 {
			s, err := flagAtIdx(mem, p*len(page)+idx)
			if err != nil {
				return s, nil
			}

			if s != "" {
				fmt.Println(s)
				return s, nil
			}
		}

		p++

		if p > 100000 {
			return "", fmt.Errorf("sanity check. have a nice day.")
		}
	}
}

func flagAtIdx(mem *Memory, idx int) (string, error) {
	end, err := mem.ReadAddress(idx + FlagStringLen - 1)
	if err != nil {
		return "", fmt.Errorf("check flag: %w", err)
	}

	if end != '}' {
		return "", nil
	}

	var b bytes.Buffer

	b.WriteString("gc24{")

	for i := idx + 5; i < idx+FlagStringLen; i++ {
		bb, err := mem.ReadAddress(i)
		if err != nil {
			return "", fmt.Errorf("check flag: %w", err)
		}
		_ = b.WriteByte(bb)
	}

	return string(b.Bytes()), nil
}
