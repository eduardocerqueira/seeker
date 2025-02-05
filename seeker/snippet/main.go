//date: 2025-02-05T17:03:37Z
//url: https://api.github.com/gists/93e8e9f945f01b79cccc7324a0a0ed00
//owner: https://api.github.com/users/CAT5NEKO

package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {
	inputFile, err := os.Open("input.txt")
	if err != nil {
		fmt.Println("Error opening input file:", err)
		return
	}
	defer inputFile.Close()

	outputFile, err := os.Create("output.txt")
	if err != nil {
		fmt.Println("Error creating output file:", err)
		return
	}
	defer outputFile.Close()

	scanner := bufio.NewScanner(inputFile)
	for scanner.Scan() {
		line := scanner.Text()
		if atIndex := strings.Index(line, "@"); atIndex != -1 {
			domain := line[atIndex+1:]
			_, err := outputFile.WriteString(domain + "\n")
			if err != nil {
				fmt.Println("Error writing to output file:", err)
				return
			}
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading input file:", err)
	}
}
