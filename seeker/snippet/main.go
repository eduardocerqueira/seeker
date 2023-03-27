//date: 2023-03-27T16:37:37Z
//url: https://api.github.com/gists/edbaffed259228fe5901928f3cf0a535
//owner: https://api.github.com/users/mrsuh

package main

import (
	"os"
	"fmt"
)

func writeToFile(filePath string, phrase []byte) {
    if _, err := os.Stat(filePath); err != nil {
	    os.Create(filePath)
	}

	file, err := os.OpenFile(filePath, os.O_RDWR, 0755)
	if err != nil {
	    panic(err)
	}
	defer file.Close()

	_, err = file.Write(phrase)
	if err != nil {
	    panic(err)
	}
	file.Sync()
}

func readFromFile(filePath string) []byte {
    file, err := os.Open(filePath)
	if err != nil {
	    panic(err)
	}
	buffer := make([]byte, 1024)
	_, err = file.Read(buffer)
	if err != nil {
	    panic(err)
	}

	return buffer
}

func main() {
    filePath := "./aes.key"
    phrase := []byte("hello world")

    writeToFile(filePath, phrase)
    newPhrase := readFromFile(filePath)

    fmt.Println(string(phrase))
    fmt.Println(string(newPhrase))
}
