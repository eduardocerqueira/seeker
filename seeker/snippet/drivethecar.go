//date: 2025-12-12T16:57:02Z
//url: https://api.github.com/gists/93279a5d907ea71a14d501ee03635614
//owner: https://api.github.com/users/0xIrshad

package main

import (
	"fmt"
	"golang.org/x/crypto/sha3"
)

var charset = []rune("abcdefghijklmnopqrstuvwxyz" +
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
	"0123456789" +
	"æèØ¾んにちは" +
	"!@#$%^&*()-_=+[]{}<>?")

func DriveTheCar(input, salt string, length int) string {
	hash := sha3.NewShake256()
	hash.Write([]byte(input + salt))

	res := make([]rune, length)
	buf := make([]byte, 2)

	for i := 0; i < length; i++ {
		hash.Read(buf)
		idx := int(buf[0]) % len(charset)
		res[i] = charset[idx]
	}

	return string(res)
}

func main() {
	fmt.Println(DriveTheCar("", "", 48))
}
